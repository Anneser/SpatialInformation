#!/usr/bin/env python3
"""
Preprocessing script for spatial information analysis.
Processes raw data files and prepares them for further analysis.
"""

import argparse
from pathlib import Path

from spatialinfo.spatial_information import (
    add_trial_column,
    load_data,
    remove_interpolated_values,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data files for analysis")
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory containing data"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for preprocessed data",
    )
    return parser.parse_args()


def preprocess_session(input_path: Path, output_path: Path):
    """Process a single experimental session."""
    # Load the data
    dff_data, behavior_data = load_data(input_path)

    # - Clean the data
    behavior_data = remove_interpolated_values(behavior_data, n_corr=2)
    behavior_data = add_trial_column(behavior_data)
    # - Standardize formats
    dff_data.columns = [f"neuron_{i}" for i in range(dff_data.shape[1])]
    # - Generate QC metrics
    # Save preprocessed data
    output_path.mkdir(parents=True, exist_ok=True)
    dff_data.to_pickle(output_path / "processed_dff.pkl")
    behavior_data.to_pickle(output_path / "processed_behavior.pkl")


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    # Process each session
    for session_dir in input_dir.glob("*"):
        if session_dir.is_dir():
            print(f"Processing session: {session_dir.name}")
            preprocess_session(session_dir, output_dir / session_dir.name)


if __name__ == "__main__":
    main()
