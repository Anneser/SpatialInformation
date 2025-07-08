"""
@author: Lukas Anneser

code prototype written by Julio Esparza,
obtained from https://github.com/PridaLab/hippocampal_manifolds

"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# from kneed import KneeLocator
# import umap
import scipy.spatial
from matplotlib.colors import Normalize

# from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


def compute_abids_make(arr, n_neigh=50, verbose=False):
    """
    Compute the Angle-Based Intrinsic Dimensionality (ABID) of a dataset.

    Parameters:
        arr (numpy.ndarray):
            Array of shape (n_samples, n_features) representing the data.
        n_neigh (int):
            Number of nearest neighbors used for computing the structure index.
        verbose (bool):
            Whether to display a progress bar.

    Returns:
        np.ndarray:
            Estimated intrinsic dimensionality for each point in the dataset.
    """

    def abid(X, k, x, search_struct, offset=1):
        """
        Computes the Angle-Based Intrinsic Dimensionality (ABID) for a given point.
        Parameters:
            X (numpy.ndarray):
                The dataset from which the point is drawn.
            k (int):
                Number of nearest neighbors to consider.
            x (numpy.ndarray):
                The point for which to compute the ABID.
            search_struct (scipy.spatial.cKDTree):
                KDTree structure for efficient neighbor search.
            offset (int):
                Offset to skip the point itself in the neighbor search.
        Returns:
            float:
                The estimated intrinsic dimensionality for the point.
        """
        neighbor_norms, neighbor_indices = search_struct.query(x, k + offset)

        # Extract neighbor coordinates and compute displacement vectors
        neighbors = X[neighbor_indices[offset:]] - x

        # Normalize the neighbor vectors
        normed_neighbors = neighbors / neighbor_norms[offset:, None]

        # Compute squared cosine similarity matrix
        para_coss = normed_neighbors.T @ normed_neighbors

        # Compute intrinsic dimensionality estimate
        return k**2 / np.sum(np.square(para_coss))

    search_struct = scipy.spatial.cKDTree(arr)

    abid_values = []
    for x in arr:
        abid_values.append(abid(arr, n_neigh, x, search_struct))

    return np.array(abid_values)


def compute_abids(arr, n_neigh=50, verbose=True):
    """
    Compute the Angle-Based Intrinsic Dimensionality (ABID) of a dataset.

    Parameters:
        arr (numpy.ndarray):
            Array of shape (n_samples, n_features) representing the data.
        n_neigh (int):
            Number of nearest neighbors used for computing the structure index.
        verbose (bool):
            Whether to display a progress bar.

    Returns:
        np.ndarray:
            Estimated intrinsic dimensionality for each point in the dataset.
    """

    def abid(X, k, x, search_struct, offset=1):
        """
        Computes the Angle-Based Intrinsic Dimensionality (ABID) for a given point.
        Parameters:
            X (numpy.ndarray):
                The dataset from which the point is drawn.
            k (int):
                Number of nearest neighbors to consider.
            x (numpy.ndarray):
                The point for which to compute the ABID.
            search_struct (scipy.spatial.cKDTree):
                KDTree structure for efficient neighbor search.
            offset (int):
                Offset to skip the point itself in the neighbor search.
        Returns:
            float:
                The estimated intrinsic dimensionality for the point.
        """
        neighbor_norms, neighbor_indices = search_struct.query(x, k + offset)

        # Extract neighbor coordinates and compute displacement vectors
        neighbors = X[neighbor_indices[offset:]] - x

        # Normalize the neighbor vectors
        normed_neighbors = neighbors / neighbor_norms[offset:, None]

        # Compute squared cosine similarity matrix
        para_coss = normed_neighbors.T @ normed_neighbors

        # Compute intrinsic dimensionality estimate
        return k**2 / np.sum(np.square(para_coss))

    search_struct = scipy.spatial.cKDTree(arr)

    abid_values = []
    for x in tqdm(arr, desc="Computing ABID", leave=False) if verbose else arr:
        abid_values.append(abid(arr, n_neigh, x, search_struct))

    return np.array(abid_values)


def plot_manifold(spatial_data, concat_emb):
    """
    Visualize the manifold embedding with spatial color coding.

    Args:
        spatial_data (np.ndarray): Array of shape (n_samples, 2)
        containing spatial coordinates (x, y).
        concat_emb (np.ndarray): Array of shape (n_samples, 3)
        containing 3D embedding coordinates.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object containing the plots.

    This function creates a 2x2 grid of 3D scatter plots:
        - The first subplot colors points by binary x-position (left/right).
        - The second subplot colors points by continuous y-position with a colorbar.
        - The third subplot colors points by time using magma colormap.
        - The fourth subplot colors points by trial, inferred from y-position resets.
    """
    # Normalize x and y separately for color mapping
    norm_x = Normalize(vmin=spatial_data[:, 0].min(), vmax=spatial_data[:, 0].max())
    norm_y = Normalize(vmin=spatial_data[:, 1].min(), vmax=spatial_data[:, 1].max())

    colors_x = cm.viridis(norm_x(spatial_data[:, 0]))  # Color by x
    colors_y = cm.plasma(norm_y(spatial_data[:, 1]))  # Color by y

    # Threshold X to assign binary groups (e.g., midpoint split)
    x_values = spatial_data[:, 0]
    x_threshold = np.mean(x_values)  # or pick a manual value
    labels_x = x_values < x_threshold  # True for "right", False for "left"

    # Define your two colors
    new_red = (255 / 255, 113 / 255, 91 / 255)
    new_teal = (147 / 255, 225 / 255, 216 / 255)

    # Map binary labels to RGB colors
    colors_rgb = np.where(labels_x[:, None], new_red, new_teal)

    # Color by index (proxy for time)
    indices = np.arange(spatial_data.shape[0])
    norm_idx = Normalize(vmin=indices.min(), vmax=indices.max())
    colors_idx = cm.magma(norm_idx(indices))

    # Infer trial numbers: new trial when y decreases
    y_vals = spatial_data[:, 1]
    trial_starts = np.where(np.diff(y_vals) < 0)[0] + 1
    trial_ids = np.zeros_like(y_vals, dtype=int)
    current_trial = 0
    last_idx = 0
    for start in np.append(trial_starts, len(y_vals)):
        trial_ids[last_idx:start] = current_trial
        current_trial += 1
        last_idx = start
    # Assign a color to each trial
    n_trials = trial_ids.max() + 1
    trial_cmap = cm.get_cmap("tab20", n_trials)
    colors_trial = trial_cmap(trial_ids)

    # Create figure and 3D subplots
    fig = plt.figure(figsize=(16, 10), facecolor="black")

    # Subplot 1: Color-coded by x-position
    ax1 = fig.add_subplot(221, projection="3d", facecolor="black")
    ax1.scatter(
        concat_emb[:, 0], concat_emb[:, 1], concat_emb[:, 2], c=colors_rgb, s=10
    )
    ax1.set_title("Color-coded by X-position")
    ax1.tick_params(colors="white")
    ax1.xaxis.label.set_color("white")
    ax1.yaxis.label.set_color("white")
    ax1.zaxis.label.set_color("white")

    # Subplot 2: Color-coded by y-position
    ax2 = fig.add_subplot(222, projection="3d", facecolor="black")
    ax2.scatter(concat_emb[:, 0], concat_emb[:, 1], concat_emb[:, 2], c=colors_y, s=10)
    ax2.set_title("Color-coded by Y-position")
    ax2.tick_params(colors="white")
    ax2.xaxis.label.set_color("white")
    ax2.yaxis.label.set_color("white")
    ax2.zaxis.label.set_color("white")
    sm_y = cm.ScalarMappable(cmap="plasma", norm=norm_y)
    sm_y.set_array([])
    cbar_y = plt.colorbar(sm_y, ax=ax2, shrink=0.5)
    cbar_y.set_label("Y Position")

    # Subplot 3: Color-coded by index (time)
    ax3 = fig.add_subplot(223, projection="3d", facecolor="black")
    ax3.scatter(
        concat_emb[:, 0], concat_emb[:, 1], concat_emb[:, 2], c=colors_idx, s=10
    )
    ax3.set_title("Color-coded by Index (Time)")
    ax3.tick_params(colors="white")
    ax3.xaxis.label.set_color("white")
    ax3.yaxis.label.set_color("white")
    ax3.zaxis.label.set_color("white")
    sm_idx = cm.ScalarMappable(cmap="magma", norm=norm_idx)
    sm_idx.set_array([])
    cbar_idx = plt.colorbar(sm_idx, ax=ax3, shrink=0.5)
    cbar_idx.set_label("Index (Time)")

    # Subplot 4: Color-coded by trial
    ax4 = fig.add_subplot(224, projection="3d", facecolor="black")
    ax4.scatter(
        concat_emb[:, 0], concat_emb[:, 1], concat_emb[:, 2], c=colors_trial, s=10
    )
    ax4.set_title("Color-coded by Trial")
    ax4.tick_params(colors="white")
    ax4.xaxis.label.set_color("white")
    ax4.yaxis.label.set_color("white")
    ax4.zaxis.label.set_color("white")
    # Create a colorbar for trials
    sm_trial = cm.ScalarMappable(
        cmap=trial_cmap, norm=Normalize(vmin=0, vmax=n_trials - 1)
    )
    sm_trial.set_array([])
    cbar_trial = plt.colorbar(sm_trial, ax=ax4, ticks=np.arange(n_trials), shrink=0.5)
    cbar_trial.set_label("Trial")
    cbar_trial.ax.set_yticklabels([str(i + 1) for i in range(n_trials)])

    plt.tight_layout()
    return fig, colors_x
