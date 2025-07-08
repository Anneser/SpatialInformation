#!/usr/bin/env python3
"""
Manifold analysis script.
Performs dimensionality reduction and manifold analysis on neural data.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from spatialinfo.dimensionality_utils import compute_abids_make, plot_manifold
from spatialinfo.spatial_information import (
    add_trial_column,
    load_data,
    remove_interpolated_values,
    temporal_binning,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Perform manifold analysis")
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
        help="Output directory for manifold results",
    )
    return parser.parse_args()


def analyze_manifold(activity_data, spatial_data, output_path: Path):
    """Perform manifold analysis for a single session."""
    abid_dim = compute_abids_make(activity_data)
    print(f"ABID: {np.nanmean(abid_dim):.2f}", end="", flush=True)

    # mom_dim = MOM().fit_transform(activity_data, n_neighbors=25)
    # print(f" | MOM: {mom_dim:.2f}", end='', flush=True)
    # tle_dim = TLE().fit_transform(activity_data, n_neighbors=25)
    # print(f" | TLE: {tle_dim:.2f}", flush=True)
    # PCA analysis
    # start by MinMax-Scaling
    scaler = MinMaxScaler()
    X = scaler.fit_transform(activity_data.T)  # Now, shape (n_samples, n_neurons)
    # fit PCA transform
    pca = PCA()
    pca.fit(X)
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    plt.figure(figsize=(5, 3))
    plt.plot(pca.explained_variance_ratio_, ".--")
    plt.xlabel("principal component")
    plt.ylabel("explained variance ratio")
    plt.savefig(output_path / "explained_variance.png")
    plt.close()

    # UMAP analysis
    model = umap.UMAP(
        n_neighbors=30, n_components=3, min_dist=0.1, metric="correlation"
    )
    model.fit(activity_data)
    concat_emb = model.transform(activity_data)

    fig, colors_x = plot_manifold(spatial_data, concat_emb)
    plt.savefig(output_path / "manifold.png")
    plt.close()

    # Create PCA figure and 3D subplot
    fig = plt.figure(figsize=(5, 4))

    # Subplot 1: Color-coded by x-position
    ax1 = fig.add_subplot(111, projection="3d")
    ax1.scatter(
        pca.components_.T[:, 0],
        pca.components_.T[:, 2],
        pca.components_.T[:, 3],
        c=colors_x,
        s=5,
    )
    plt.savefig(output_path / "PCA.png")
    plt.close()

    # Generate plots
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.savefig(output_path / "pca_variance.png")
    plt.close()

    return np.nanmean(abid_dim)


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each session
    results = {}
    for session_dir in input_dir.glob("*"):
        if session_dir.is_dir():
            print(f"\nAnalyzing manifold: {session_dir.name}\n")

            # Load preprocessed data
            # Load the data
            dff_data, behavior_data = load_data(session_dir)
            behavior_data = remove_interpolated_values(behavior_data, n_corr=2)
            behavior_data = add_trial_column(behavior_data)
            # dff_data = pd.read_pickle(session_dir / "processed_dff.pkl")
            # dff_data.columns = [f"{i}" for i in range(dff_data.shape[1])]
            # behavior_data = pd.read_pickle(session_dir / "processed_behavior.pkl")
            activity_data, spatial_data = temporal_binning(
                dff_data, behavior_data, sec_per_bin=0.5, only_moving=False
            )

            # Run analysis
            session_output = output_dir / session_dir.name
            session_output.mkdir(exist_ok=True)
            dimensionality = analyze_manifold(
                activity_data, spatial_data, session_output
            )

            results[session_dir.name] = dimensionality

    # Save summary results
    pd.Series(results).to_csv(output_dir / "dimensionality_summary.csv")


if __name__ == "__main__":
    main()
