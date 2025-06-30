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

    This function creates a 1x2 grid of 3D scatter plots:
        - The first subplot colors points by binary x-position (left/right).
        - The second subplot colors points by continuous y-position with a colorbar.
    """
    # Normalize x and y separately for color mapping
    norm_x = Normalize(vmin=spatial_data[:, 0].min(), vmax=spatial_data[:, 0].max())
    norm_y = Normalize(vmin=spatial_data[:, 1].min(), vmax=spatial_data[:, 1].max())

    colors_x = cm.viridis(norm_x(spatial_data[:, 0]))  # Color by x
    colors_y = cm.plasma(norm_y(spatial_data[:, 1]))  # Color by y

    # Filter points where x = 0 and x ≠ 0
    # mask_x0 = spatial_data[:, 0] == 0
    # mask_x_nonzero = spatial_data[:, 0] != 0

    # Threshold X to assign binary groups (e.g., midpoint split)
    x_values = spatial_data[:, 0]
    x_threshold = np.mean(x_values)  # or pick a manual value
    labels_x = x_values < x_threshold  # True for "right", False for "left"

    # Define your two colors
    new_red = (255 / 255, 113 / 255, 91 / 255)
    new_teal = (147 / 255, 225 / 255, 216 / 255)

    # Map binary labels to RGB colors
    colors_rgb = np.where(labels_x[:, None], new_red, new_teal)

    # Create figure and 3D subplots
    fig = plt.figure(figsize=(10, 5), facecolor="black")

    # Subplot 1: Color-coded by x-position
    ax1 = fig.add_subplot(121, projection="3d", facecolor="black")
    ax1.scatter(
        concat_emb[:, 0], concat_emb[:, 1], concat_emb[:, 2], c=colors_rgb, s=10
    )
    ax1.set_title("Color-coded by X-position")
    # Set white for all axis lines and ticks
    ax1.tick_params(colors="white")  # Ticks
    ax1.xaxis.label.set_color("white")
    ax1.yaxis.label.set_color("white")
    ax1.zaxis.label.set_color("white")

    # Subplot 2: Color-coded by y-position
    ax2 = fig.add_subplot(122, projection="3d", facecolor="black")
    ax2.scatter(concat_emb[:, 0], concat_emb[:, 1], concat_emb[:, 2], c=colors_y, s=10)
    ax2.set_title("Color-coded by Y-position")
    ax2.tick_params(colors="white")  # Ticks
    ax2.xaxis.label.set_color("white")
    ax2.yaxis.label.set_color("white")
    ax2.zaxis.label.set_color("white")
    # Add colorbar for y
    sm_y = cm.ScalarMappable(cmap="plasma", norm=norm_y)
    sm_y.set_array([])
    cbar_y = plt.colorbar(sm_y, ax=ax2, shrink=0.5)
    cbar_y.set_label("Y Position")

    return fig, colors_x
