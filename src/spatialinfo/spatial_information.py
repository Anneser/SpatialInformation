import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.feature_selection import mutual_info_classif
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from keras_tuner import HyperModel, RandomSearch


def load_data(file_path: Path):
    """
    Load neuronal activity and behavioral data from a given directory.

    This function searches the specified directory for files ending in
    '*dff.pkl' and '*behavior.pkl', loads them as pandas DataFrames,
    and returns them.

    Parameters:
        file_path (Path): Path to the directory containing the .pkl files.

    Returns:
        tuple: A tuple containing:
            - dff (pd.DataFrame): DataFrame with neuronal activity (e.g., ΔF/F traces).
            - behavior (pd.DataFrame): DataFrame with behavioral data
              (e.g., positions, timestamps).
    """
    dff = pd.read_pickle(sorted(file_path.glob("*dff.pkl"))[0])
    behavior = pd.read_pickle(sorted(file_path.glob("*behavior.pkl"))[0])

    return dff, behavior


def remove_interpolated_values(behavior, n_corr=2):
    """
    Cleans up interpolated X values by reassigning them to the nearest valid corridor
    value in time order.

    Parameters:
        behavior (pandas.core.frame.DataFrame): contains X,
            Y columns with spatial information
        n_corr (int): number of corridors to consider (defaults to 2)

    Returns:
        behavior (pandas.core.frame.DataFrame): cleaned behavior data with
            X values reassigned
    """
    # Get the most common corridor values
    corr_val = behavior["X"].value_counts().head(n_corr).index.values

    # Replace invalid X values with the next valid value in the time sequence
    last_valid_value = None
    for i in range(len(behavior)):
        if behavior.loc[i, "X"] not in corr_val:
            if last_valid_value is not None:
                behavior.loc[i, "X"] = last_valid_value
        else:
            last_valid_value = behavior.loc[i, "X"]

    return behavior


def binning(dff, behavior, n_bins=30, n_corr=2, fps=30, bins=None):
    """
    Bins occupancy and activity data. Assumes that corridors have been
    completely sampled by the animal and that distinct corridors are
    characterized by X position.

    Args:
        dff (pandas.DataFrame): df/f calcium imaging data (rows: time; columns: neurons)
        behavior (pandas.DataFrame): contains X, Y columns with spatial information and frdIn with tail vigor
        n_bins (int): number of bins, defaults to 30
        n_corr (int): number of corridors, defaults to 2
        fps (int): frames per second, defaults to 30
        bins (numpy.ndarray): edges for bins can optionally be provided

    Returns:
        time_per_bin (pandas.core.frame.DataFrame): time spent in each bin, n_bins x n_corridors
        summed_traces (pandas.core.frame.DataFrame): summed activity for each bin, n_bins x n_corridors x n_neurons
        bins (numpy.ndarray or IntervalIndex): returns the computed bins
    """
    if bins is None:
        # Create bin labels, cut the Y positions into bins and X into corridor values.
        behavior.loc[:, "Y_bin"], bins = pd.cut(
            behavior["Y"], bins=n_bins, labels=False, retbins=True
        )
    else:
        behavior.loc[:, "Y_bin"] = pd.cut(behavior["Y"], bins=bins, labels=False)
    # behavior.loc[:,'X_bin'] = pd.cut(behavior['X'], bins=len(behavior.X.unique()))

    # Group by the corridor ID (X position) and Y-bin, and calculate the time per bin
    time_per_bin = (
        behavior.groupby(["X", "Y_bin"], observed=False).size().unstack(fill_value=0)
    )

    # Automatically generate column names based on the number of corridors
    num_corridors = time_per_bin.shape[0]
    column_names = [f"Corridor_{i + 1}" for i in range(num_corridors)]
    time_per_bin = pd.DataFrame(time_per_bin.T.values, columns=column_names) / fps

    # Sum calcium traces for each bin and corridor
    summed_traces = (
        behavior[["X", "Y_bin"]]
        .join(dff, how="left")
        .groupby(["X", "Y_bin"])
        .sum()
        .unstack(level=0, fill_value=0)
    )

    # Make column and row names intuitive:
    summed_traces.rename_axis("Space bin", inplace=True)
    summed_traces.columns.set_names(["Neuron", "Corridor"], inplace=True)
    # summed_traces.convert_dtypes().dtypes # hard conversion
    summed_traces.apply(pd.to_numeric).dtypes

    return time_per_bin, summed_traces, bins


def avg_activity(time_per_bin, summed_traces):
    """
    Devides summed neural activity matrix by the occupancy matrix to obtain the average
    activity in each spatial bin.

    Args:
        time_per_bin (pandas.core_frame.DataFrame): time spent in each bin, n_bins x n_corridors
        summed_traces (pandas.core.frame.DataFrame): summed activity per bin, bins x corridors x neurons
    Returns:
        avg_activity_mtx (pandas.core_frame.DataFrame): average activity in each spatial bin.
    """
    # Create an empty DataFrame with the same structure as summed_traces
    avg_activity_mtx = pd.DataFrame(
        index=summed_traces.index, columns=summed_traces.columns
    )

    # Iterate through each neuron column group (level 1 in the MultiIndex)
    for neuron in summed_traces.columns.get_level_values("Neuron").unique():
        trace = summed_traces[neuron].apply(pd.to_numeric)
        # divide by time occupancy for each neuron and assign to new DataFrame
        avg_activity_mtx[neuron] = trace.div(time_per_bin.values, axis=0)

    return avg_activity_mtx


def spatial_info_calc(avg_activity_mtx, time_per_bin):
    """
    Calculates the spatial information for each neuron based on the average activity in
    spatial bins and occupancy probability.

        Parameters:
            avg_activity_mtx (pandas.core_frame.DataFrame): average activity in each
                spatial bin (MultiIndex: Neuron, Corridor).
            time_per_bin (pandas.core_frame.DataFrame): time spent in each spatial bin
                (n_bins x n_corridors).

        Returns:
            spatial_info (pandas.Series): spatial information for each neuron,
                indexed by neuron ID.
            spatial_spec (pandas.Series): spatial specificity for each neuron,
                indexed by neuron ID.
    """
    P_x = time_per_bin / time_per_bin.sum().sum()
    # Initialize a Series to store spatial information for each neuron
    spatial_info = pd.Series(
        index=avg_activity_mtx.T.index.get_level_values("Neuron").unique(), dtype=float
    )
    spatial_spec = spatial_info.copy(deep=True)
    # Iterate through each neuron
    for neuron in spatial_info.index:
        neuron_activity = avg_activity_mtx.T.loc[neuron]
        lambda_avg = (neuron_activity * P_x.T.values).sum().sum()
        data = neuron_activity / lambda_avg
        valid_data = data[data > 0].dropna()
        info = np.nansum(neuron_activity * np.log2(valid_data) * P_x.T.values)
        spatial_info[neuron] = info
        spatial_spec[neuron] = info / lambda_avg

    return spatial_info, spatial_spec


def spec_z_score(dff, behavior, sampling_rate=30, n_permut=1000, n_bins=10):
    """
    Calculates specificity z-score by circularly permutating activity & behavioral data.
    From this shifted data, a null distribution of specificity scores is computed.
    The specificity z-score is then calculated by subtracting the mean of this
    distribution and dividing by its standard deviation.

    Parameters:
        dff (DataFrame): dF/F activity data.
        behavior (DataFrame): Behavioral data to be shifted.
        sampling_rate (int): Sampling rate of the data (default: 30).
        n_permut (int): Number of permutations (default: 1000).
        n_bins (int): Number of spatial bins (default: 10).

    Returns:
        actual_spatial_spec (float): Actual specificity score.
        shifted_spec (list): List of specificity scores from permuted data.
    """
    # Create an empty list to store specificity scores
    shifted_spec = []

    # Iterate through shifts, creating a null distribution
    for i in range(int(-(n_permut // 2)), int((n_permut // 2) + 1)):
        shift_amount = (i * sampling_rate) // 2  # 500 ms shifts

        # Safely circularly shift the behavior data
        shift_amount = shift_amount % len(behavior)
        if shift_amount != 0:
            shifted_behavior = pd.concat(
                [behavior.iloc[-shift_amount:], behavior.iloc[:-shift_amount]]
            ).reset_index(drop=True)
        else:
            shifted_behavior = behavior.copy()

        # Calculate specificity score for the shifted data
        time_per_bin, summed_traces, _ = binning(dff, shifted_behavior, n_bins=n_bins)
        avg_act_mtx = avg_activity(time_per_bin, summed_traces)
        spatial_info, spatial_spec = spatial_info_calc(avg_act_mtx, time_per_bin)
        shifted_spec = [
            spec
            for spec in shifted_spec
            if not spec.isna().any() and not np.isinf(spec).any()
        ]

        # Collect shifted specificity scores, exclude shift_amount = 0
        if shift_amount != 0:
            shifted_spec.append(spatial_spec)
        else:
            actual_spatial_spec = spatial_spec

    return (actual_spatial_spec - pd.DataFrame(shifted_spec).mean()) / pd.DataFrame(
        shifted_spec
    ).std()


def pop_z_score(spatial_spec):
    """
    Compute the population-wide z-score for a spatial tuning curve.

    This function normalizes a pandas Series by subtracting the mean
    and dividing by the standard deviation across the entire population.
    If the standard deviation is zero, returns NaNs to avoid division by zero.

    Args:
        spatial_spec (pd.Series): A Series representing spatial tuning
                                  for a single neuron or condition.

    Returns:
        pd.Series: Z-scored values of the input Series. If the standard
                   deviation is zero, returns a Series of NaNs.
    """
    std_dev = spatial_spec.std()
    if std_dev == 0:
        warnings.warn("Standard deviation is zero; returning NaNs for z-score.")
        return pd.Series(np.nan, index=spatial_spec.index)
    return (spatial_spec - spatial_spec.mean()) / std_dev


def trials_over_time(dff, behavior, n_bins=30, verbose=True):
    """
    Returns a multi-index DataFrame containing the average activity per spatial bin
    per corridor per trial for each neuron.

    Parameters:
        dff (DataFrame): dF/F activity data in timepoints (rows) x neurons (columns).
        behavior (DataFrame): Behavioral data in timepoints (rows) x behavior (column).
        n_bins (int): Number of spatial bins to divide the Y-position.

    Returns:
        DataFrame: Multi-index DataFrame with (neuron, trial, corridor, Y_bin) as index.
    """
    # Validate input data
    is_valid, message = validate_data(dff, behavior, verbose=verbose)
    if not is_valid:
        raise ValueError(f"Data validation failed: {message}")

    # Step 1: Extract metadata
    n_neurons = dff.shape[1]

    # Step 2: Create spatial bins for Y-position
    behavior["Y_bin"] = pd.cut(behavior["Y"], bins=n_bins, labels=False)

    # Step 3: Identify unique trials and corridors
    trials = behavior["trial"].unique()
    corridors = behavior["X"].unique()

    # Step 4: Initialize storage for results
    results = []

    # Step 5: Iterate over neurons, trials, and corridors
    for neuron in range(n_neurons):
        neuron_activity = dff.iloc[:, neuron]  # Select the neuron

        for trial in trials:
            trial_data = behavior[behavior["trial"] == trial]
            trial_activity = neuron_activity.loc[
                trial_data.index
            ]  # Activity for this trial

            for corridor in corridors:
                corridor_data = trial_data[trial_data["X"] == corridor]
                corridor_activity = trial_activity.loc[corridor_data.index]

                # Compute average activity per spatial bin
                avg_activity_per_bin = corridor_activity.groupby(
                    corridor_data["Y_bin"]
                ).mean()

                # Ensure all bins are represented (fill missing bins with NaN)
                avg_activity_per_bin = avg_activity_per_bin.reindex(
                    range(n_bins), fill_value=np.nan
                )

                # Store results
                for y_bin, activity in avg_activity_per_bin.items():
                    results.append([neuron, trial, corridor, y_bin, activity])

    # Step 6: Convert results to DataFrame
    results_df = pd.DataFrame(
        results, columns=["neuron", "trial", "corridor", "Y_bin", "activity"]
    )

    # Step 7: Pivot to create a multi-dimensional DataFrame

    results_df = results_df.pivot_table(
        index=["neuron", "trial", "corridor", "Y_bin"], values="activity"
    )

    # Step 8: Normalize activity per neuron (Min/Max scaling across all bins and trials)
    # scaler = MinMaxScaler()
    # results_df[:] = scaler.fit_transform(results_df) # scale later
    return results_df


def trials_over_time_figure(
    neuron_ID, dff, behavior, n_bins=10, cmap="magma", export_figure=False
):
    """
    For any given neuron ID, this plots the activity per spatial bin per corridor
    per trial in two heatmaps with shared color scales and consistent x-axis
    (spatial bins).

    Parameters:
        neuron_ID (int): column number of neuron in dff.
        dff (DataFrame): dF/F activity data in timepoints (rows) x neurons (columns).
        behavior (DataFrame): Behavioral data in timepoints (rows) x behavior (column).
        n_bins (int): number of spatial bins to plot (defaults to 10).
        export_figure (boolean): defines whether plot should be saved
            (defaults to False).
        cmap (str): defines colormap, defaults to "magma".
    """

    # Step 1: Extract neuron activity
    neuron_activity = dff.iloc[:, neuron_ID]

    # Step 2: Create spatial bins for the Y-position
    behavior["Y_bin"] = pd.cut(behavior["Y"], bins=n_bins, labels=False)

    # Step 3: Separate trials and corridors
    trials = behavior["trial"].unique()
    corridors = behavior["X"].unique()

    # Create a dictionary to store binned activity for each corridor
    binned_activity = {corridor: [] for corridor in corridors}

    # Step 4: Iterate over trials and bin activity for each corridor
    for trial in trials:
        trial_data = behavior[behavior["trial"] == trial]
        trial_activity = neuron_activity.loc[trial_data.index]

        for corridor in corridors:
            corridor_data = trial_data[trial_data["X"] == corridor]
            corridor_activity = trial_activity.loc[corridor_data.index]

            # Bin the activity by spatial bins
            avg_activity_per_bin = corridor_activity.groupby(
                corridor_data["Y_bin"]
            ).mean()

            # Ensure all spatial bins (0 to n_bins-1) are present
            avg_activity_per_bin = avg_activity_per_bin.reindex(
                range(n_bins), fill_value=np.nan
            )
            binned_activity[corridor].append(avg_activity_per_bin)

    # Step 5: Convert lists to DataFrames for heatmap plotting
    heatmap_data_corridor_1 = pd.DataFrame(binned_activity[corridors[0]])
    heatmap_data_corridor_2 = pd.DataFrame(binned_activity[corridors[1]])

    # Step 6: Determine shared color scale limits
    vmin = min(heatmap_data_corridor_1.min().min(), heatmap_data_corridor_2.min().min())
    vmax = max(heatmap_data_corridor_1.max().max(), heatmap_data_corridor_2.max().max())

    # Step 7: Plot heatmaps with shared color scale and consistent x-axis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    sns.heatmap(
        heatmap_data_corridor_1, cmap=cmap, ax=axes[0], vmin=vmin, vmax=vmax, cbar=True
    )
    axes[0].set_title(f"Corridor {corridors[0]}")
    axes[0].set_xlabel("Spatial Bin")
    axes[0].set_ylabel("Trial")
    axes[0].set_xticks(np.arange(n_bins))
    axes[0].set_xticklabels(np.arange(n_bins))

    sns.heatmap(
        heatmap_data_corridor_2, cmap=cmap, ax=axes[1], vmin=vmin, vmax=vmax, cbar=True
    )
    axes[1].set_title(f"Corridor {corridors[1]}")
    axes[1].set_xlabel("Spatial Bin")
    axes[1].set_xticks(np.arange(n_bins))
    axes[1].set_xticklabels(np.arange(n_bins))

    plt.tight_layout()
    if export_figure:
        plt.savefig(f"neuron_{neuron_ID}_plot.svg")
    plt.show()


def add_trial_column(behavior):
    """
    Adds a 'trial' column to the behavior DataFrame.

    A trial starts when:
    1. The animal moves backward (Y-position decreases by being teleported to
        starting position).
    2. The corridor changes (X-position changes).
    Trials with fewer than 30 timepoints are merged with the previous trial.

    Parameters:
        behavior (DataFrame): Must contain 'X' (corridor) and 'Y'
            (position in corridor).

    Returns:
        behavior (DataFrame): Updated DataFrame with a new 'trial' column.
    """
    MIN_TRIAL_LENGTH = 30
    trial_counter = 0
    trial_numbers = [0]  # First trial is always 0
    trial_start_idx = 0  # Track start of current trial

    # Iterate through the behavior data (starting from index 1)
    for i in range(1, len(behavior)):
        # Get change in Y and change in X
        y_diff = behavior.loc[i, "Y"] - behavior.loc[i - 1, "Y"]
        x_diff = behavior.loc[i, "X"] - behavior.loc[i - 1, "X"]

        # Check if a new trial should start
        if y_diff < -1 or x_diff != 0:  # -1 protects against positional jitter
            # Check length of previous trial
            trial_length = i - trial_start_idx

            if trial_length >= MIN_TRIAL_LENGTH:
                trial_counter += 1  # Increment trial counter
                trial_start_idx = i
            # If trial too short, keep previous trial number (merging trials)

        # Append the trial number for the current row
        trial_numbers.append(trial_counter)

    # Assign trial numbers to the DataFrame
    behavior["trial"] = trial_numbers

    return behavior


def load_anatomy(file_path: Path):
    anatomy_path = sorted(file_path.glob("*_anatomy.tif"))[0]
    mask_path = sorted(file_path.glob("*_masks.png"))[0]
    # correlation_map_path = sorted(file_path.glob("*_cm.tif"))[0]
    anatomy = np.array(Image.open(anatomy_path))
    mask = np.array(Image.open(mask_path))
    # correlation_map = np.array(Image.open(correlation_map_path))
    return anatomy, mask  # , correlation_map


def plot_neurons(file_path: Path, neuron_index: np.ndarray, spec_df, save_figure=False):
    """
    Plots the spatial specificity of bona fide place cells in the anatomical context,
    with three subplots:
    1. Spatial specificity
    2. Specificity Z-score
    3. Population Z-score

    Parameters:
        file_path (Path): path to the folder containing all files for a given dataset.
        neuron_index (array): list of indices for place cells.
        spec_df (DataFrame): DataFrame with columns "spatial_specificity",
            "specificity_zscore", and "population_zscore".
        save_figure (boolean): Whether the figure should just be printed or also saved.
    Returns:
        None
    """
    # Load the anatomical data
    anatomy, mask = load_anatomy(file_path)

    # Prepare figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharex=True, sharey=True)
    subplot_titles = [
        "Spatial Specificity",
        "Specificity Z-Score",
        "Population Z-Score",
    ]
    columns = ["spatial_specificity", "specificity_zscore", "population_zscore"]
    color_maps = ["inferno", "inferno", "inferno"]  # plasma, viridis,...

    # Iterate over subplots
    for ax, title, col, cmap in zip(axes, subplot_titles, columns, color_maps):
        # Plot the anatomical correlation map as background
        ax.imshow(anatomy, cmap="gray", aspect="auto")
        ax.set_title(title)
        ax.axis("off")

        # Get the values for color-coding
        values = spec_df.loc[neuron_index, col]

        # Normalize the values for color mapping
        norm = plt.Normalize(values.min(), values.max())

        # Plot each neuron, color-coded by the column value
        for neuron_id, value in zip(neuron_index, values):
            # Get the mask coordinates for the current neuron
            y_coords, x_coords = np.where(
                mask == neuron_id + 1
            )  # Mask indices are 1-based

            # Plot the neuron with the color corresponding to its value
            if len(y_coords) > 0:
                ax.scatter(
                    x_coords, y_coords, s=5, color=plt.get_cmap(cmap)(norm(value))
                )
                # Draw an outline around the neuron
                neuron_mask = (mask == neuron_id + 1).astype(int)
                ax.contour(neuron_mask, levels=[0.5], colors="white", linewidths=1.5)

        # Add a color bar to the subplot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

    # Adjust layout
    plt.tight_layout()

    # Save or show the figure
    if save_figure:
        save_path = file_path / "place_cells_spatial_specificity_subplots.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        save_path_svg = file_path / "place_cells_spatial_specificity_subplots.svg"
        plt.savefig(save_path_svg)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()


def format_for_ann(df):
    """
    Converts the multi-index DataFrame into input-output
    format for training a neural network.

    Ensures that each row represents one spatial bin (with trial & corridor info),
    and columns represent neuronal activity.

    Parameters:
        df (DataFrame): Multi-index DataFrame with (neuron, trial, corridor, Y_bin) index.

    Returns:
        X (ndarray): Feature matrix where each row is a neuronal activity vector (neurons as columns).
        y (tuple): Tuple of target labels (bins, corridors).
    """
    # Reset index so we can manipulate it
    df = df.reset_index()

    # Pivot so that each row represents a (trial, corridor, bin),
    # and each column is a neuron
    reshaped_df = df.pivot_table(
        index=["trial", "corridor", "Y_bin"], columns="neuron", values="activity"
    )

    # Drop any remaining NaNs (bins with missing neuron activity)
    reshaped_df = reshaped_df.dropna()

    # Extract the final feature matrix
    X = reshaped_df.values  # Now, shape (n_samples, n_neurons)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Extract labels
    y_bins = reshaped_df.index.get_level_values("Y_bin").values  # Bin labels
    y_corridors = reshaped_df.index.get_level_values(
        "corridor"
    ).values  # Corridor labels

    # Map corridor labels to categorical values (0 and 1)
    unique_corridors = sorted(np.unique(y_corridors))
    corridor_mapping = {unique_corridors[0]: 0, unique_corridors[1]: 1}
    y_corridors = np.array([corridor_mapping[c] for c in y_corridors])

    return X, (y_bins, y_corridors)


def proximity_weighted_accuracy(y_true, y_pred, max_distance=2):
    """
    Compute accuracy where predictions closer to the true bin get partial credit.

    Parameters:
        y_true (array-like): True bin labels.
        y_pred (array-like): Predicted bin labels.
        max_distance (int): Maximum distance for partial credit.

    Returns:
        float: Proximity-weighted accuracy score.
    """
    total_score = 0
    for true, pred in zip(y_true, y_pred):
        distance = abs(true - pred)
        if distance <= max_distance:
            total_score += 1 - (distance / (max_distance + 1))

    return total_score / len(y_true)


def place_field_size(avg_activity_mtx, neuron_ID, n_bins=30, chamber_size=150):  # FIXME
    """
    Calculates the place fields of bona fide place cells and returns their center of
    mass as well as their size.

    Parameters:
        avg_activity_mtx (DataFrame): Average activity in each spatial bin
            (MultiIndex: Neuron, Corridor).
        neuron_ID (array): List of bona fide place cells to be used for indexing.
        n_bins (int): Number of spatial bins (default: 30).
        chamber_size (int): Size of the chamber in VR units (default: 150).

    Returns:
        pf_list (list of dicts): List of dictionaries with the size and
            center of mass of the place fields for each neuron.
    """
    pf_list = []

    # Iterate over each neuron in the provided list
    for neuron in neuron_ID:
        neuron_data = avg_activity_mtx.T.xs(neuron, level="Neuron")

        # Step 1: Determine the peak activity (95th percentile) for the neuron
        peak_activity = np.percentile(neuron_data.values, 95)

        # Step 2: Define the place field threshold (80% of the peak activity)
        threshold = 0.8 * peak_activity

        # Step 3: Identify place fields as contiguous regions above the threshold
        place_fields = []
        for corridor in neuron_data.T.columns:
            activity = neuron_data[corridor].values
            above_threshold = activity > threshold

            # Find contiguous bins above the threshold
            place_field_bins = []
            current_field = []
            for idx, is_above in enumerate(above_threshold):
                if is_above:
                    current_field.append(idx)
                elif current_field:
                    # Append and reset the current field
                    place_field_bins.append(current_field)
                    current_field = []

            # Check the last field
            if current_field:
                place_field_bins.append(current_field)

            # Step 4: Calculate the COM and size for each place field
            for pf in place_field_bins:
                pf_size = len(pf) / n_bins * chamber_size
                # if pf_size < 0.3 * chamber_size:  # Filter by size
                # (less than 30% of chamber size)
                # pf_com = center_of_mass(activity[pf])
                place_fields.append({"size": pf_size})

        # Step 5: Store the results for the current neuron
        pf_list.append({"neuron": neuron, "place_fields": place_fields})

    return pf_list


def temporal_binning(dff, behavior, sec_per_bin=1, fps=30, only_moving=True):
    """
    Bins occupancy and activity data into temporal bins.
    Assumes that distinct corridors are characterized by X position and
    that a trial column exists in behavior

    Parameters:
        dff (pandas.core.frame.DataFrame): df/f calcium imaging data
            (rows: time; columns: neurons)
        behavior (pandas.core.frame.DataFrame): contains X, Y columns with
            spatial information, trial column and frdIn with tail vigor
        sec_per_bin (int): how many seconds will be averaged into one bin,
            defaults to 1.
        fps (int): recording speed in frames per second, defaults to 30
        only_moving (bool): removing frames in which the animal is not moving,
            defaults to True.

    Returns:
        activity_data (np.ndarray): activity matrix after binning,
            columns = features, rows = bins
        spatial_data (np.ndarray): average position of animal during each bin,
            columns = X, Y, rows = bins
    """
    # Compute the number of frames per bin
    frames_per_bin = sec_per_bin * fps

    # remove frames in which the animal is not moving
    if only_moving:
        id_mask = behavior.Y.diff().values != 0
        dff = dff.loc[id_mask].reset_index()
        behavior = behavior.loc[id_mask].reset_index()
    # Determine number of bins
    num_bins = int(np.ceil(len(dff) / frames_per_bin))

    # Initialize lists to store binned data
    binned_activity = []
    binned_spatial = []

    for i in range(num_bins):
        start_idx = math.floor(i * frames_per_bin)
        end_idx = math.ceil(min((i + 1) * frames_per_bin, len(dff)))

        activity_bin = dff.iloc[start_idx:end_idx].mean(axis=0).values
        spatial_bin = behavior[["X", "Y"]].iloc[start_idx:end_idx].mean(axis=0).values

        if np.any(np.diff(behavior.trial.iloc[start_idx:end_idx].values) != 0):
            continue
        else:
            binned_activity.append(activity_bin)
            binned_spatial.append(spatial_bin)

    # Convert lists to numpy arrays
    activity_data = np.array(binned_activity).astype(float)
    spatial_data = np.array(binned_spatial).astype(float)

    return activity_data, spatial_data


def plot_distribution(spec_z_score_values, pop_z_score_values, spatial_spec):
    """Plot the distribution of specificity and population z-scores."""
    # Create figure and GridSpec layout
    fig = plt.figure(figsize=(9, 3))
    gs = fig.add_gridspec(1, 2)  # One rows, two columns

    # Subplot 1 (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(spec_z_score_values, pop_z_score_values, "o")
    ax1.axhline(y=1, color="r", linestyle="--")
    ax1.axvline(x=1, color="r", linestyle="--")
    ax1.set_xlabel("Specificity z-score")
    ax1.set_ylabel("Population z-score")
    ax1.set_title("Specificity vs. Population Z-score")

    # Subplot 2 (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(spatial_spec, bins=15, color="skyblue", edgecolor="black")
    ax2.set_xlabel("Specificity Score [bits per activity unit]")
    ax2.set_ylabel("Number of Neurons in Bin")
    ax2.set_title("Distribution of Specificity Scores")
    plt.tight_layout()
    return fig


def calculate_neural_stats(dff_data, behavior_data):
    """Calculate basic statistics of neural activity."""
    time_per_bin, summed_traces, _ = binning(dff_data, behavior_data)
    avg_act_mtx = avg_activity(time_per_bin, summed_traces)
    _, spatial_spec = spatial_info_calc(avg_act_mtx, time_per_bin)

    spec_z_score_values = spec_z_score(dff_data, behavior_data, n_permut=100, n_bins=30)
    pop_z_score_values = pop_z_score(spatial_spec)
    return spec_z_score_values, pop_z_score_values, avg_act_mtx, spatial_spec


def validate_data(dff_data, behavior_data, verbose=True):
    """
    Validate input data for analysis functions.

    Args:
        dff_data: DataFrame containing neural activity
        behavior_data: DataFrame containing behavioral data
        verbose: Whether to print validation info (default: True)

    Returns:
        tuple: (is_valid, error_message)
    """
    if verbose:
        print("\nData Validation:")
        print("-" * 50)
        print(f"DFF data shape: {dff_data.shape}")
        print(f"Behavior data shape: {behavior_data.shape}")
        print(f"DFF data types:\n{dff_data.dtypes}")
        print(f"Behavior data types:\n{behavior_data.dtypes}")

        # Check for NaN values
        dff_nans = dff_data.isna().sum().sum()
        behav_nans = behavior_data.isna().sum().sum()
        print(f"\nNaN values in DFF data: {dff_nans}")
        print(f"NaN values in behavior data: {behav_nans}")

        # Check for infinite values
        dff_infs = np.isinf(dff_data.astype(float).values).sum()
        print(f"Infinite values in DFF data: {dff_infs}")

    # Check for data validity
    if dff_data.shape[0] != behavior_data.shape[0]:
        return False, "DFF and behavior data have different numbers of samples"

    if dff_nans > 0:
        return False, f"Found {dff_nans} NaN values in DFF data"

    if behav_nans > 0:
        return False, f"Found {behav_nans} NaN values in behavior data"

    if dff_infs > 0:
        return False, f"Found {dff_infs} infinite values in DFF data"

    return True, "Data validation passed"
