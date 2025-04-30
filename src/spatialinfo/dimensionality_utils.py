"""
@author: Lukas Anneser

code prototype written by Julio Esparza, 
obtained from https://github.com/PridaLab/hippocampal_manifolds

"""

from tqdm import tqdm
import numpy as np 
from kneed import KneeLocator
import umap
import scipy.spatial 


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
        """Computes the Angle-Based Intrinsic Dimensionality (ABID) for a given point."""
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
