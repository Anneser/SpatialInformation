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

from spatialinfo.dimensionality_utils import calculate_dimensionality


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


def analyze_manifold(dff_data, behavior_data, output_path: Path):
    """Perform manifold analysis for a single session."""
    # PCA analysis
    pca = PCA()
    pca_result = pca.fit_transform(dff_data)

    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # UMAP analysis
    reducer = umap.UMAP()
    umap_result = reducer.fit_transform(dff_data)

    # Calculate intrinsic dimensionality
    dim = calculate_dimensionality(dff_data)

    # Save results
    np.save(output_path / "pca_embeddings.npy", pca_result)
    np.save(output_path / "umap_embeddings.npy", umap_result)
    np.save(output_path / "pca_explained_variance.npy", explained_variance)

    # Generate plots
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.savefig(output_path / "pca_variance.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(umap_result[:, 0], umap_result[:, 1], alpha=0.5)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.savefig(output_path / "umap_embedding.png")
    plt.close()

    return dim


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each session
    results = {}
    for session_dir in input_dir.glob("*"):
        if session_dir.is_dir():
            print(f"Analyzing manifold: {session_dir.name}")

            # Load preprocessed data
            dff_data = pd.read_pickle(session_dir / "processed_dff.pkl")
            behavior_data = pd.read_pickle(session_dir / "processed_behavior.pkl")

            # Run analysis
            session_output = output_dir / session_dir.name
            session_output.mkdir(exist_ok=True)
            dimensionality = analyze_manifold(dff_data, behavior_data, session_output)

            results[session_dir.name] = dimensionality

    # Save summary results
    pd.Series(results).to_csv(output_dir / "dimensionality_summary.csv")


if __name__ == "__main__":
    main()
