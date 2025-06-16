#!/usr/bin/env python3
"""
Statistical analysis of neural activity and behavioral data.
Generates summary statistics and distributions.
"""

import argparse
from pathlib import Path

# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def calculate_neural_stats(dff_data):
    """Calculate basic statistics of neural activity."""
    stats = {
        "mean_activity": dff_data.mean(),
        "std_activity": dff_data.std(),
        "max_activity": dff_data.max(),
        "sparseness": (dff_data > 0).mean(),
    }
    return pd.DataFrame(stats)


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
            # behavior_data = pd.read_pickle(session_dir / "processed_behavior.pkl")

            # Calculate statistics
            neural_stats = calculate_neural_stats(dff_data)
            all_neural_stats.append(neural_stats)

            # Generate plots
            plt.figure(figsize=(10, 6))
            sns.histplot(data=dff_data.mean(), bins=50)
            plt.title(f"Distribution of Mean Neural Activity - {session_dir.name}")
            plt.savefig(output_dir / f"{session_dir.name}_activity_dist.png")
            plt.close()

    # Save combined statistics
    combined_stats = pd.concat(all_neural_stats)
    combined_stats.to_csv(output_dir / "neural_statistics.csv")


if __name__ == "__main__":
    main()
