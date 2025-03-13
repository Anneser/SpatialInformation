"""
@author: Lukas Anneser

code prototype written by Julio Esparza, 
obtained from https://github.com/PridaLab/hippocampal_manifolds

"""

# FIXME - installing libraries breaks env
from tqdm import tqdm
import numpy as np 
#from kneed import KneeLocator
#import umap
import spatialinfo.validation as dimval 
import scipy.spatial 


def compute_abids(arr, n_neigh=50, verbose=True):
    """
    Compute the Adaptive Ball-Based Intrinsic Dimensionality (ABID) of a dataset.

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
        """Computes the Adaptive Ball-Based Intrinsic Dimensionality (ABID) for a given point."""
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


def compute_umap_dim(X, n_neigh = 5, max_dim = 10):
    '''
    Estimate the data's dimensionality using UMAP trust and cont (see Venna, 
    Jarkko, and Samuel Kaski. Local multidimensional scaling with controlled 
    tradeoff between trustworthiness and continuity." Proceedings of 5th 
    Workshop on Self-Organizing Maps. 2005.)
    
    Parameters
    ----------
    X : 2D array
        n_samples x n_features data
    
    n_neigh: int 
        number of neighbours used to compute trustworthiness.

    max_dim: int
        maximum dimension contemplated

    Returns
    -------
    estimated dimensionality
    '''
    max_dim = np.min([max_dim, X.shape[1]])
    rank_X  = dimval.compute_rank_indices(X)
    trust_num = np.zeros((max_dim,))*np.nan
    cont_num = np.zeros((max_dim,))*np.nan

    for dim in range(np.min([max_dim, X.shape[1]])):
        model = umap.UMAP(n_neighbors = n_neigh, n_components =dim+1)
        emb = model.fit_transform(X)

        #2. Compute trustworthiness
        temp = dimval.trustworthiness_vector(X, emb, n_neigh, 
                                                indices_source = rank_X)
        trust_num[dim] = temp[-1]
        #2. Compute continuity
        temp = dimval.continuity_vector(X, emb, n_neigh)
        cont_num[dim] = temp[-1]

    dim_space = np.arange(1,max_dim+1).astype(int)  
    kl = KneeLocator(dim_space, trust_num, curve = "concave", 
                                                direction = "increasing")
    if kl.knee:
        trust_dim = kl.knee
    else:
        trust_dim = np.nan
    kl = KneeLocator(dim_space, cont_num, curve = "concave", 
                                                direction = "increasing")
    if kl.knee:
        cont_dim = kl.knee
    else:
        cont_dim = np.nan

    hmean_dim = (2*trust_dim*cont_dim)/(trust_dim+cont_dim)
    return trust_num, trust_dim, cont_num, cont_dim, hmean_dim