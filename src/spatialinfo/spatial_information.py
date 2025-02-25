import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


def load_data(file_path: Path):
    dff = pd.read_pickle(sorted(file_path.glob("*dff.pkl"))[0])
    behavior = pd.read_pickle(sorted(file_path.glob("*behavior.pkl"))[0])
    return dff, behavior


def remove_interpolated_values(behavior, n_corr=2):
    """
    Cleans up interpolated X values by reassigning them to the nearest valid corridor value in time order.

            Parameters:
                    behavior (pandas.core.frame.DataFrame): contains X, Y columns with spatial information
                    n_corr (int): number of corridors to consider (defaults to 2)

            Returns:
                    behavior (pandas.core.frame.DataFrame): cleaned behavior data with X values reassigned
    """
    # Get the most common corridor values
    corr_val = behavior['X'].value_counts().head(n_corr).index.values

    # Replace invalid X values with the next valid value in the time sequence
    last_valid_value = None
    for i in range(len(behavior)):
        if behavior.loc[i, 'X'] not in corr_val:
            if last_valid_value is not None:
                behavior.loc[i, 'X'] = last_valid_value
        else:
            last_valid_value = behavior.loc[i, 'X']

    return behavior


def binning(dff, behavior, n_bins=30, n_corr=2, fps=30, bins=None):
    """
    Bins occupancy and activity data. Assumes that corridors have been completely sampled by the animal.
    Assumes that distinct corridors are characterized by X position.

            Parameters:
                    dff (pandas.core.frame.DataFrame): df/f calcium imaging data (rows: time; columns: neurons)
                    behavior (pandas.core.frame.DataFrame): contains X, Y columns with spatial information and frdIn with tail vigor
                    n_bins (int): number of bins, defaults to 30
                    n_corr (int): number of corridors, defaults to 2
                    fps (int): frames per second, defaults to 30
                    bins (numpy.ndarray): edges for bins can optionally be provided

            Returns:
                    time_per_bin (pandas.core.frame.DataFrame): time spent in each bin, n_bins x n_corridors
                    summed_traces (pandas.core.frame.DataFrame): summed activity for each bin, n_bins x n_corridors x n_neurons
                    bins (numpy.ndarray or IntervalIndex): returns the computed bins
    """
    if bins == None:
        # Create bin labels, cut the Y positions into bins and X into corridor values.
        behavior['Y_bin'], bins = pd.cut(behavior['Y'], bins=n_bins, labels=False, retbins=True)
    else:
        behavior['Y_bin'] = pd.cut(behavior['Y'], bins=bins, labels=False)
    behavior['X_bin'] = pd.cut(behavior['X'], bins=len(behavior.X.unique()))

    # Group by the corridor ID (X position) and Y-bin, and calculate the time per bin
    time_per_bin = behavior.groupby(['X_bin', 'Y_bin'], observed=False).size().unstack(fill_value=0)

    # Automatically generate column names based on the number of corridors
    num_corridors = time_per_bin.shape[0]
    column_names = [f"Corridor_{i + 1}" for i in range(num_corridors)]
    time_per_bin = pd.DataFrame(time_per_bin.T.values, columns=column_names) / fps

    # Sum calcium traces for each bin and corridor
    summed_traces = (
        behavior[['X', 'Y_bin']]
        .join(dff, how='left')
        .groupby(['X', 'Y_bin'])
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
    Devides summed neural activity matrix by the occupancy matrix to obtain the average activity in each spatial bin.

            Parameters:
                    time_per_bin (pandas.core_frame.DataFrame): time spent in each bin, n_bins x n_corridors
                    summed_traces (pandas.core.frame.DataFrame): summed activity for each bin, n_bins x n_corridors x n_neurons
            Returns:
                    avg_activity_mtx (pandas.core_frame.DataFrame): average activity in each spatial bin.
    """
    # Create an empty DataFrame with the same structure as summed_traces
    avg_activity_mtx = pd.DataFrame(index=summed_traces.index, columns=summed_traces.columns)

    # Iterate through each neuron column group (level 1 in the MultiIndex)
    for neuron in summed_traces.columns.get_level_values("Neuron").unique():
        trace = summed_traces[neuron].apply(pd.to_numeric)
        # Apply the division by the time occupancy for each neuron and assign to the new DataFrame
        avg_activity_mtx[neuron] = (
            trace.div(time_per_bin.values, axis=0)
        )

    return avg_activity_mtx


def spatial_info_calc(avg_activity_mtx, time_per_bin):
    """
    Calculates the spatial information for each neuron based on the average activity in spatial bins and occupancy probability.

        Parameters:
            avg_activity_mtx (pandas.core_frame.DataFrame): average activity in each spatial bin (MultiIndex: Neuron, Corridor).
            time_per_bin (pandas.core_frame.DataFrame): time spent in each spatial bin (n_bins x n_corridors).

        Returns:
            spatial_info (pandas.Series): spatial information for each neuron, indexed by neuron ID.
            spatial_spec (pandas.Series): spatial specificity for each neuron, indexed by neuron ID.
    """
    P_x = time_per_bin / time_per_bin.sum().sum()
    # Initialize a Series to store spatial information for each neuron
    spatial_info = pd.Series(index=avg_activity_mtx.T.index.get_level_values("Neuron").unique(), dtype=float)
    spatial_spec = spatial_info.copy(deep=True)
    # Iterate through each neuron
    for neuron in spatial_info.index:
        neuron_activity = avg_activity_mtx.T.loc[neuron]
        lambda_avg = (neuron_activity * P_x.T.values).sum().sum()
        data = neuron_activity / lambda_avg
        valid_data = data[data > 0].dropna()
        I = np.nansum(neuron_activity * np.log2(valid_data) * P_x.T.values)
        spatial_info[neuron] = I
        spatial_spec[neuron] = I / lambda_avg

    return spatial_info, spatial_spec


def spec_z_score(dff, behavior, sampling_rate=30, n_permut=1000, n_bins=10):
    """
    Calculates specificity z-score by circularly permutating activity & behavioral data. From this shifted data,
    a null distribution of specificity scores is computed. The specificity z-score is then calculated by subtracting
    the mean of this distribution and dividing by its standard deviation.

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
            shifted_behavior = pd.concat([behavior.iloc[-shift_amount:], behavior.iloc[:-shift_amount]]).reset_index(
                drop=True)
        else:
            shifted_behavior = behavior.copy()

        # Calculate specificity score for the shifted data
        time_per_bin, summed_traces = binning(dff, shifted_behavior, n_bins=n_bins)
        avg_act_mtx = avg_activity(time_per_bin, summed_traces)
        spatial_info, spatial_spec = spatial_info_calc(avg_act_mtx, time_per_bin)
        shifted_spec = [spec for spec in shifted_spec if not spec.isna().any() and not np.isinf(spec).any()]

        # Collect shifted specificity scores, exclude shift_amount = 0
        if shift_amount != 0:
            shifted_spec.append(spatial_spec)
        else:
            actual_spatial_spec = spatial_spec

    return (actual_spatial_spec - pd.DataFrame(shifted_spec).mean()) / pd.DataFrame(shifted_spec).std()


def pop_z_score(spatial_spec):
    std_dev = spatial_spec.std()
    if std_dev == 0:
        return pd.Series(np.nan, index=spatial_spec.index)
    return (spatial_spec - spatial_spec.mean()) / std_dev


def trials_over_time(neuron_ID, dff, behavior, n_bins=10, export_figure=False):
    '''
    For any given neuron ID, this plots the activity per spatial bin per corridor per trial in two heatmaps
    with shared color scales and consistent x-axis (spatial bins).

    Parameters:
        neuron_ID (int): column number of neuron in dff.
        dff (DataFrame): dF/F activity data in timepoints (rows) x neurons (columns).
        behavior (DataFrame): Behavioral data in timepoints (rows) x behavior (column).
        n_bins (int): number of spatial bins to plot (defaults to 10).
        export_figure (boolean): defines whether plot should be saved (defaults to False).
    '''

    # Step 1: Extract neuron activity
    neuron_activity = dff.iloc[:, neuron_ID]

    # Step 2: Create spatial bins for the Y-position
    behavior['Y_bin'] = pd.cut(behavior['Y'], bins=n_bins, labels=False)

    # Step 3: Separate trials and corridors
    trials = behavior['trial'].unique()
    corridors = behavior['X'].unique()

    # Create a dictionary to store binned activity for each corridor
    binned_activity = {corridor: [] for corridor in corridors}

    # Step 4: Iterate over trials and bin activity for each corridor
    for trial in trials:
        trial_data = behavior[behavior['trial'] == trial]
        trial_activity = neuron_activity.loc[trial_data.index]

        for corridor in corridors:
            corridor_data = trial_data[trial_data['X'] == corridor]
            corridor_activity = trial_activity.loc[corridor_data.index]

            # Bin the activity by spatial bins
            avg_activity_per_bin = corridor_activity.groupby(corridor_data['Y_bin']).mean()

            # Ensure all spatial bins (0 to n_bins-1) are present
            avg_activity_per_bin = avg_activity_per_bin.reindex(range(n_bins), fill_value=np.nan)
            binned_activity[corridor].append(avg_activity_per_bin)

    # Step 5: Convert lists to DataFrames for heatmap plotting
    heatmap_data_corridor_1 = pd.DataFrame(binned_activity[corridors[0]])
    heatmap_data_corridor_2 = pd.DataFrame(binned_activity[corridors[1]])

    # Step 6: Determine shared color scale limits
    vmin = min(heatmap_data_corridor_1.min().min(), heatmap_data_corridor_2.min().min())
    vmax = max(heatmap_data_corridor_1.max().max(), heatmap_data_corridor_2.max().max())

    # Step 7: Plot heatmaps with shared color scale and consistent x-axis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    sns.heatmap(heatmap_data_corridor_1, cmap='magma', ax=axes[0], vmin=vmin, vmax=vmax, cbar=True)
    axes[0].set_title(f"Corridor {corridors[0]}")
    axes[0].set_xlabel("Spatial Bin")
    axes[0].set_ylabel("Trial")
    axes[0].set_xticks(np.arange(n_bins))
    axes[0].set_xticklabels(np.arange(n_bins))

    sns.heatmap(heatmap_data_corridor_2, cmap='magma', ax=axes[1], vmin=vmin, vmax=vmax, cbar=True)
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
    Adds a "trial" column to the behavior DataFrame based on Y-position resets (teleports).

    Parameters:
        behavior (DataFrame): Behavioral data containing the Y column (position).

    Returns:
        behavior (DataFrame): Updated DataFrame with a new "trial" column.
    """
    # Initialize the trial counter and create an empty list for trial numbers
    trial_counter = 0
    trial_numbers = []

    # Iterate through the Y-position data
    for i in range(len(behavior)):
        # Check if the current Y position is 0 and the previous Y position was non-zero
        if i > 0 and behavior.loc[i, 'Y'] == 0 and behavior.loc[i - 1, 'Y'] != 0:
            # Increment the trial counter
            trial_counter += 1

        # Assign the current trial number
        trial_numbers.append(trial_counter)

    # Add the trial column to the DataFrame
    behavior['trial'] = trial_numbers

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
    Plots the spatial specificity of bona fide place cells in the anatomical context, with three subplots:
    1. Spatial specificity
    2. Specificity Z-score
    3. Population Z-score

    Parameters:
        file_path (Path): path to the folder containing all files for a given dataset.
        neuron_index (array): list of indices for place cells.
        spec_df (DataFrame): DataFrame with columns "spatial_specificity", "specificity_zscore", and "population_zscore".
        save_figure (boolean): Whether the figure should just be printed or also saved.
    Returns:
        None
    """
    # Load the anatomical data
    anatomy, mask = load_anatomy(file_path)

    # Prepare figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharex=True, sharey=True)
    subplot_titles = ["Spatial Specificity", "Specificity Z-Score", "Population Z-Score"]
    columns = ["spatial_specificity", "specificity_zscore", "population_zscore"]
    color_maps = ["inferno", "inferno", "inferno"]  # plasma, viridis,...

    # Iterate over subplots
    for ax, title, col, cmap in zip(axes, subplot_titles, columns, color_maps):
        # Plot the anatomical correlation map as background
        ax.imshow(anatomy, cmap='gray', aspect='auto')
        ax.set_title(title)
        ax.axis("off")

        # Get the values for color-coding
        values = spec_df.loc[neuron_index, col]

        # Normalize the values for color mapping
        norm = plt.Normalize(values.min(), values.max())

        # Plot each neuron, color-coded by the column value
        for neuron_id, value in zip(neuron_index, values):
            # Get the mask coordinates for the current neuron
            y_coords, x_coords = np.where(mask == neuron_id + 1)  # Mask indices are 1-based

            # Plot the neuron with the color corresponding to its value
            if len(y_coords) > 0:
                ax.scatter(x_coords, y_coords, s=5, color=plt.get_cmap(cmap)(norm(value)))
                # Draw an outline around the neuron
                neuron_mask = (mask == neuron_id + 1).astype(int)
                ax.contour(neuron_mask, levels=[0.5], colors='white', linewidths=1.5)

        # Add a color bar to the subplot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

    # Adjust layout
    plt.tight_layout()

    # Save or show the figure
    if save_figure:
        save_path = file_path / "place_cells_spatial_specificity_subplots.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        save_path_svg = file_path / "place_cells_spatial_specificity_subplots.svg"
        plt.savefig(save_path_svg)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()
