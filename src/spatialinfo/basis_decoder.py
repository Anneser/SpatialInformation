
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

        # Standardize each neuron's activity across all bins and corridors
        standardized_act_mtx = avg_act_mtx.copy()
        standardized_loocv_bins = loocv_bins.copy()
        # neuron_stats = {}  # Store means and stds for later normalization
        for neuron in range(dff.shape[1]):
            neuron_mean = avg_act_mtx.loc[:, avg_act_mtx.columns.get_level_values("Neuron")==neuron].values.mean()
            neuron_std = avg_act_mtx.loc[:, avg_act_mtx.columns.get_level_values("Neuron")==neuron].values.std()
            standardized_act_mtx.loc[:, standardized_act_mtx.columns.get_level_values("Neuron")==neuron] = (avg_act_mtx.loc[:, avg_act_mtx.columns.get_level_values("Neuron")==neuron] - neuron_mean) / neuron_std
            standardized_loocv_bins.loc[:, standardized_loocv_bins.columns.get_level_values("Neuron")==neuron] = (loocv_bins.loc[:, loocv_bins.columns.get_level_values("Neuron")==neuron] - neuron_mean) / neuron_std
                
        # Get the single corridor value for this trial
        # Check if there's only one unique X value, otherwise take the most common one
        if len(bh_test["X"].unique()) == 1:
            corridor = bh_test["X"].unique()[0]
        else:
            corridor = bh_test["X"].mode()[0]
                
        # Loop through each space bin in the test trial
        for space_bin in range(n_bins):
            if space_bin not in standardized_loocv_bins.index:
                continue  # Skip if test data does not contain this bin

            # Extract the population vector for this bin
            pop_vector = standardized_loocv_bins.xs(key=corridor, level="Corridor", axis=1).loc[space_bin]
                
            # Get the 30 most active cells in this bin using standardized values
            top_30_neurons = pop_vector.nlargest(30)
                
            # Calculate relative activity (how much more active compared to others)
            relative_activity = top_30_neurons #/ top_30_neurons.mean()

            # Get corresponding activity maps from standardized templates
            scaled_matrix = standardized_act_mtx[top_30_neurons.index].multiply(relative_activity, axis=1, level="Neuron")

            # Sum across neurons for each corridor separately
            decoded_map = pd.DataFrame()
            for corr in bh["X"].unique():
                corridor_matrix = scaled_matrix.loc[:, scaled_matrix.columns.get_level_values("Corridor") == corr]
                decoded_map[corr] = corridor_matrix.sum(axis=1)

            # Get the most active bin and its corresponding corridor
            row_idx, col_idx = np.unravel_index(decoded_map.values.argmax(), decoded_map.shape)
            decoded_bin = int(decoded_map.index[row_idx])
            decoded_corridor = int(decoded_map.columns[col_idx])

            # Store results
            decoded_positions.append({
            "trial": trial,
            "true_corridor": corridor,
            "decoded_corridor": decoded_corridor,
            "true_bin": space_bin,
            "decoded_bin": decoded_bin
            })

    return pd.DataFrame(decoded_positions)