
from spatialinfo.spatial_information import remove_interpolated_values, binning, avg_activity, add_trial_column
import numpy as np
import pandas as pd




def fixed_template(dff, behavior, n_corridors=2, n_bins=30):
    """
    Implements direct basis decoding with LOOCV using fixed binning.

    Parameters:
        dff (DataFrame): Calcium imaging traces for all neurons.
        behavior (DataFrame): Behavioral data with X (corridor), Y (position).
        n_corridors (int): Number of corridors in the setting (default=2).
        n_bins (int): Number of spatial bins (default=30).

    Returns:
        decoded_position (DataFrame): Decoded X and Y positions.
    """
    # Preprocess behavior data
    bh = remove_interpolated_values(behavior, n_corr=n_corridors)
    if "trial" not in bh.columns:
        bh = add_trial_column(bh)

    # Initialize storage for decoded results
    decoded_positions = []

    # Perform Leave-One-Out Cross-Validation (LOOCV)
    for trial in bh["trial"].unique():
        print(f"Performing LOOCV on trial {trial}.")

        # Split data into training and test sets
        dff_test = dff[bh["trial"] == trial]
        dff_train = dff[bh["trial"] != trial]
        bh_test = bh[bh["trial"] == trial]
        bh_train = bh[bh["trial"] != trial]

        # Compute binning for training data
        time_per_bin, summed_traces, bins = binning(dff_train, bh_train, n_bins=n_bins)

        # Compute average activity templates (fixed decoder)
        avg_act_mtx = avg_activity(time_per_bin, summed_traces)

        # Apply the same binning to test data
        time_per_bin, summed_traces, _ = binning(dff_test, bh_test, n_bins=n_bins, bins=bins)
        loocv_bins = avg_activity(time_per_bin, summed_traces)
        
        # Get the single corridor value for this trial
        corridor = bh_test["X"].iloc[0]

        # Loop through each space bin in the test trial
        for space_bin in range(n_bins):
            if space_bin not in loocv_bins.index:
                continue  # Skip if test data does not contain this bin

            # Extract the population vector for this bin
            pop_vector = loocv_bins.xs(key=corridor, level="Corridor", axis=1).loc[space_bin]

            # Compute weighted activity map
            scaled_matrix = avg_act_mtx.xs(key=corridor, level="Corridor", axis=1).mul(pop_vector, axis=1)

            # Decode by summing across neurons
            decoded_map = scaled_matrix.sum(axis=1)

            # Store results
            decoded_positions.append({
                "trial": trial,
                "corridor": corridor,
                "true_bin": space_bin,
                "decoded_bin": decoded_map.idxmax()  # Pick the most active decoded bin
            })

    return pd.DataFrame(decoded_positions)