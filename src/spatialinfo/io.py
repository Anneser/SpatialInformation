from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


def save_neural_stats(
    spec_z_score_values, pop_z_score_values, avg_act_mtx, spatial_spec, output_dir
):
    """
    Save neural statistics using HDF5 format

    Args:
        spec_z_score_values: 1D array of specificity z-scores
        pop_z_score_values: 1D array of population z-scores
        avg_act_mtx: 2D array of average activity matrices
        spatial_spec: 1D array of spatial specificity values
        output_dir: Path to output directory

    Returns:
        Path: Path to saved HDF5 file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_dir / "neural_stats.h5", "w") as f:
        stats_group = f.create_group("statistics")
        matrices_group = f.create_group("matrices")

        stats_group.create_dataset(
            "specificity_zscores", data=spec_z_score_values, compression="gzip"
        )
        stats_group.create_dataset(
            "population_zscores", data=pop_z_score_values, compression="gzip"
        )
        stats_group.create_dataset(
            "spatial_specificity", data=spatial_spec, compression="gzip"
        )

        matrices_group.create_dataset(
            "average_activity", data=avg_act_mtx, compression="gzip"
        )

        f.attrs["created"] = np.bytes_(datetime.now().isoformat())
        f.attrs["total_neurons"] = len(spec_z_score_values)

    selective_idx = np.where((spec_z_score_values > 1) & (pop_z_score_values > 1))[0]
    np.save(output_dir / "selective_neurons.npy", selective_idx)

    return output_dir / "neural_stats.h5"


def load_neural_stats(file_path):
    """
    Load neural statistics from HDF5 file

    Args:
        file_path: Path to HDF5 file
    Returns:
        dict: Dictionary containing all arrays and metadata
    """
    with h5py.File(file_path, "r") as f:
        return {
            "specificity_zscores": f["statistics/specificity_zscores"][:],
            "population_zscores": f["statistics/population_zscores"][:],
            "spatial_specificity": f["statistics/spatial_specificity"][:],
            "average_activity": f["matrices/average_activity"][:],
            "created": f.attrs["created"].decode(),
            "total_neurons": f.attrs["total_neurons"],
        }
