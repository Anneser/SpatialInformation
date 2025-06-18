#!/usr/bin/env python3
"""
Statistical analysis of neural activity and behavioral data.
Generates summary statistics and distributions.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import spatialinfo.spatial_information as si
from spatialinfo.io import save_neural_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate statistical metrics")
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
        help="Output directory for statistical results",
    )
    return parser.parse_args()


def calculate_behavioral_stats(behavior_data):
    """Calculate statistics of behavioral variables."""
    # Add your behavioral analysis here
    pass


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each session
    all_neural_stats = []
    for session_dir in input_dir.glob("*"):
        if session_dir.is_dir():
            print(f"Analyzing session: {session_dir.name}")

            # Load preprocessed data
            dff_data = pd.read_pickle(session_dir / "processed_dff.pkl")
            behavior_data = pd.read_pickle(session_dir / "processed_behavior.pkl")

            # Calculate statistics
            (spec_z_scores, pop_z_scores, avg_act_mtx, spatial_spec) = (
                si.calculate_neural_stats(dff_data, behavior_data)
            )
            # Save results using the io module
            save_neural_stats(
                spec_z_scores,
                pop_z_scores,
                avg_act_mtx,
                spatial_spec,
                output_dir / f"{session_dir.name}_neural_stats.pkl",
            )
            # Directly append the statistics as a DataFrame or dict
            session_stats = pd.DataFrame(
                {
                    "session": [session_dir.name],
                    "spec_z_scores_mean": [spec_z_scores.mean()],
                    "pop_z_scores_mean": [pop_z_scores.mean()],
                    "spatial_spec_mean": [spatial_spec.mean()],
                }
            )
            all_neural_stats.append(session_stats)
            si.plot_distribution(spec_z_scores, pop_z_scores, spatial_spec)
            plt.title(f"Distribution of Mean Neural Activity - {session_dir.name}")
            plt.savefig(output_dir / f"{session_dir.name}_activity_dist.png")
            plt.close()

    # Save combined statistics
    combined_stats = pd.concat(all_neural_stats)
    combined_stats.to_csv(output_dir / "neural_statistics.csv")


if __name__ == "__main__":
    main()
