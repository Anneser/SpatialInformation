#!/usr/bin/env python3
"""
Spatial decoding analysis script.
Performs decoding of spatial information from neural activity.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from spatialinfo.decoder import SpatialDecoder


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
    # Prepare data
    X = dff_data.values
    y = behavior_data[
        "location"
    ].values  # Assuming location is encoded in behavior data

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train decoder
    decoder = SpatialDecoder()
    decoder.fit(X_train, y_train)

    # Evaluate
    y_pred = decoder.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save results
    results = {"accuracy": accuracy, "predictions": y_pred, "true_values": y_test}
    np.save(output_path / "decoding_results.npy", results)

    return accuracy


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each session
    results = {}
    for session_dir in input_dir.glob("*"):
        if session_dir.is_dir():
            print(f"Decoding session: {session_dir.name}")

            # Load preprocessed data
            dff_data = pd.read_pickle(session_dir / "processed_dff.pkl")
            behavior_data = pd.read_pickle(session_dir / "processed_behavior.pkl")

            # Run decoding
            session_output = output_dir / session_dir.name
            session_output.mkdir(exist_ok=True)
            accuracy = decode_session(dff_data, behavior_data, session_output)

            results[session_dir.name] = accuracy

    # Save summary results
    pd.Series(results).to_csv(output_dir / "decoding_accuracy_summary.csv")


if __name__ == "__main__":
    main()
