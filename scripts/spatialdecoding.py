#!/usr/bin/env python3
"""
Spatial decoding analysis script.
Performs decoding of spatial information from neural activity.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import parallel_backend
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import spatialinfo.spatial_information as si
from spatialinfo.decoder import (
    build_keras_model,
    clean_data,
    create_templates,
    match_templates,
    proximity_weighted_accuracy,
)

# from tensorflow.keras import models


def parse_args():
    parser = argparse.ArgumentParser(description="Perform spatial decoding")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing preprocessed data",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for decoding results",
    )
    return parser.parse_args()


def decode_session(dff_data, behavior_data, output_path: Path):
    """Perform decoding analysis for a single session."""
    # Set number of cores explicitly to avoid detection issues
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())

    with parallel_backend("loky", n_jobs=-1):
        # Ensure data is in the correct format
        if not isinstance(dff_data, pd.DataFrame) or not isinstance(
            behavior_data, pd.DataFrame
        ):
            print("Error: dff_data and behavior_data must be pandas DataFrames")
            return None

        if dff_data.empty or behavior_data.empty:
            print("Error: dff_data or behavior_data is empty")
            return None

        # Check for NaN values
        if dff_data.isnull().values.any() or behavior_data.isnull().values.any():
            print("Error: NaN values found in input data")
            return None
        try:
            # Prepare data with validation
            results_df = si.trials_over_time(
                dff_data, behavior_data, n_bins=30, verbose=True
            )

            if results_df is None or results_df.empty:
                print("Error: No valid results obtained from trials_over_time")
                return None

            X, (y_bins, y_corridors) = si.format_for_ann(results_df)

            # Print data types for debugging
            print("\nData types after formatting:")
            if isinstance(X, pd.DataFrame):
                print("X data types:\n", X.dtypes)
            else:
                print("X type:", type(X))

            # Convert to numpy array if needed
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()
            elif isinstance(X, pd.Series):
                X = X.to_numpy()

            # Additional validation for neural network input
            if not np.issubdtype(X.dtype, np.number):
                print(f"Error: Data is not numeric (dtype: {X.dtype})")
                return None

            if np.isnan(X).any():
                print("Error: NaN values found in formatted data")
                return None

            if np.isfinite(X).all() is False:  # This is safer than using isinf
                print("Error: Infinite values found in formatted data")
                return None

        except ValueError as e:
            print(f"Validation error: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error during decoding: {str(e)}")
            return None

        # If we get here, data is valid, proceed with splitting
        (
            X_train,
            X_test,
            y_bins_train,
            y_bins_test,
            y_corridors_train,
            y_corridors_test,
        ) = train_test_split(
            X, y_bins, y_corridors, test_size=0.1, random_state=42, stratify=y_bins
        )
        print("Training ANN...")
        # Build and train the model
        keras_model = build_keras_model(input_size=X.shape[1], n_bins=30, n_corridors=2)
        keras_model.fit(
            X_train,
            {"bin_output": y_bins_train, "corridor_output": y_corridors_train},
            epochs=200,
            batch_size=64,
        )
        # Get model predictions on the test set
        y_pred_bins, y_pred_corridors = keras_model.predict(X_test)

        # Convert probabilities to class labels
        # Get the highest probability bin
        y_pred_bins = np.argmax(y_pred_bins, axis=1)
        # Get highest probability corridor
        y_pred_corridors = np.argmax(y_pred_corridors, axis=1)

        # Evaluate proximity-weighted accuracy
        proximity_acc = proximity_weighted_accuracy(y_bins_test, y_pred_bins)
        print(f"Proximity-Weighted Accuracy: {proximity_acc:.4f}")

        # Compute confusion matrices
        conf_matrix_bins = confusion_matrix(y_bins_test, y_pred_bins)
        conf_matrix_corridors = confusion_matrix(y_corridors_test, y_pred_corridors)

        # Save confusion matrices
        np.save(output_path / "confusion_matrix_bins.npy", conf_matrix_bins)
        np.save(output_path / "confusion_matrix_corridors.npy", conf_matrix_corridors)
        # Save model
        keras_model.save(output_path / "spatial_decoder_model.keras")

        # save figure of confusion matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Bin classification confusion matrix
        sns.heatmap(conf_matrix_bins, annot=True, fmt="d", cmap="Blues", ax=ax1)
        ax1.set_xlabel("Predicted Bin")
        ax1.set_ylabel("True Bin")
        ax1.set_title("Bin Classification")

        # Corridor classification confusion matrix
        sns.heatmap(
            conf_matrix_corridors,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[0, 1],
            yticklabels=[0, 1],
            ax=ax2,
        )
        ax2.set_xlabel("Predicted Corridor")
        ax2.set_ylabel("True Corridor")
        ax2.set_title("Corridor Classification")

        plt.tight_layout()
        plt.savefig(output_path / "confusion_matrices.png")
        plt.close()

        # Save accuracy results
        accuracy = {
            "proximity_weighted_accuracy": proximity_acc,
            "accuracy_bins": accuracy_score(y_bins_test, y_pred_bins),
            "accuracy_corridors": accuracy_score(y_corridors_test, y_pred_corridors),
        }
        pd.Series(accuracy).to_csv(output_path / "decoding_accuracy.csv")

        print("Computing mutual information and information gain...")
        y_combined = y_bins + y_corridors * (max(y_bins) + 1)
        # Calculate Mutual Information using mutual_info_regression
        mutual_info = mutual_info_regression(X, y_combined)
        info_gain = mutual_info_classif(X, y_combined)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(mutual_info, ax=ax1)
        ax1.set_title("Mutual Information Distribution")
        ax1.set_xlabel("Mutual Information")
        ax1.set_ylabel("Density")

        sns.histplot(info_gain, ax=ax2)
        ax2.set_title("Information Gain Distribution")
        ax2.set_xlabel("Information Gain")
        ax2.set_ylabel("Density")
        plt.tight_layout()
        plt.savefig(output_path / "mutual_info_info_gain.png")
        plt.close()

        # Save mutual information and information gain
        pd.Series(mutual_info).to_csv(output_path / "mutual_information.csv")
        pd.Series(info_gain).to_csv(output_path / "information_gain.csv")

        # print("Employing Direct Basis Decoder...")
        # df = fixed_template(dff_data, behavior_data)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3),
        # gridspec_kw={"width_ratios": [3, 1]})

        # ax1.set_title("Basis Decoder with place cells")
        # ax1.plot(df["true_bin"], ".", lw=1, label="True")
        # ax1.plot(df["decoded_bin"], ".", lw=1, label="Decoded")
        # ax1.set_xlabel("Time (bins)")
        # ax1.set_ylabel("Bin")
        # ax1.legend()
        # ax2.plot([0, 30], [0, 30], "--", color="red")
        # ax2.scatter(df["true_bin"], df["decoded_bin"], alpha=0.2)
        # ax2.set_title("Decoded vs True Bins")
        # ax2.set_xlabel("True Bin")
        # ax2.set_ylabel("Decoded Bin")
        # plt.tight_layout()
        # plt.savefig(output_path / "basis_decoder_results.png")
        # plt.close()

        print("Performing template matching...")

        # prepare data and create templates
        dff_clean, bh_clean = clean_data(dff_data.to_numpy(), behavior_data.to_numpy())
        features, bin_values, _ = create_templates(dff_clean, bh_clean)
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features, bin_values, test_size=0.1, random_state=42, stratify=bin_values
        )

        confmatrix_dict = {}
        for method in ["correlation", "cosine", "euclidean"]:
            print("matching templates using {} metric...".format(method))
            confmatrix_tm = np.zeros((len(np.unique(y_train)), len(np.unique(y_train))))
            for vector, bin in zip(X_test, y_test):
                pred_bin = match_templates(
                    X_train, y_train, vector, distance_metric=method
                )
                confmatrix_tm[bin - 1, pred_bin - 1] += 1
            confmatrix_dict[method] = confmatrix_tm

        # plotting results
        methods = ["correlation", "cosine", "euclidean"]
        fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)

        for i in range(3):
            sns.heatmap(
                confmatrix_dict[methods[i]], fmt="d", cmap="gist_gray", ax=ax.flat[i]
            )
            ax.flat[i].title.set_text(methods[i])
            ax.flat[i].set_xlabel("Predicted Bin")
            ax.flat[i].set_ylabel("True Bin")
        plt.tight_layout()
        plt.savefig(output_path / "template_matching_results.png")
        plt.close()

        print("Decoding session completed.")


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each session
    for session_dir in input_dir.glob("*"):
        if session_dir.is_dir():
            print(f"\nProcessing session: {session_dir.name}")

            # Load preprocessed data
            try:
                dff_data = pd.read_pickle(session_dir / "processed_dff.pkl")
                behavior_data = pd.read_pickle(session_dir / "processed_behavior.pkl")
            except FileNotFoundError as e:
                print(f"Error loading data for session {session_dir.name}: {e}")
                continue

            # Create session output directory
            session_output = output_dir / session_dir.name
            session_output.mkdir(exist_ok=True)

            # Run decoding
            try:
                decode_session(dff_data, behavior_data, session_output)
                print(f"Successfully completed session: {session_dir.name}")
            except Exception as e:
                print(f"Error processing session {session_dir.name}: {e}")


if __name__ == "__main__":
    main()
