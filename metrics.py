"""
Metrics for model evaluation.

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
from sklearn.utils.graph import graph_shortest_path
from sklearn.metrics import pairwise, mean_squared_error
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr, pearsonr
from manifold_clustering import select_neighborhood
from intrinsic_dim import mide
from util import create_graph, get_k_range
        
def smape(X_true, X_pred):
    '''
    Symmetric mean absolute percentage error
    Reference:
    "O'Connor, M., & Lawrence, M. (1998). Judgmental forecasting and the use of available information. 
    Forecasting with judgment. Chichester: Wiley, 65-90."
    '''
    return np.mean(np.abs((X_true - X_pred) / (np.clip(np.abs(X_true) + np.abs(X_pred), np.finfo(float).eps, np.inf))))
    
def get_geo_dist(X, adaptive_neighbor=False, K='auto', local_intr_dim=False, verbose=0):
    
    m = X.shape[0]
    
    if adaptive_neighbor:
        
        if local_intr_dim:
            local_dims = mide(X)[1]
        else:
            local_dims = mide(X)[0]
            
        nbrs = select_neighborhood(X, local_dims, verbose=2) # list of neighborhood indices
        
        A = np.zeros((m, m)) # adjacency matrix
        D = np.zeros((m, m)) # adjacency matrix with pairwise distance
        for i in range(m):
            A[i, nbrs[i]] = np.ones((len(nbrs[i]),))
            for nbr in nbrs[i]:
                D[i, nbr] = np.linalg.norm(X[i] - X[nbr])
        
        G = graph_shortest_path(D, directed=False) # shortest path graph
        
    else:
        
        if K == 'auto':
            k_min, k_max = get_k_range(X)
            K = k_max
        
        G = create_graph(X, K, verbose=verbose)
            
    return G
    
def geo_dist_inconsistency(X, F, X_precomputed=False, verbose=0):
    ''' Geodesic distance inconsistency '''
    if X_precomputed:
        geo_X = X # geodesic distance metrix for X
    else:
        geo_X = get_geo_dist(X, verbose=verbose)
    geo_X[geo_X==0] = np.inf # if two points are not connected
    np.fill_diagonal(geo_X, 0)
    
    dist_F = pairwise.pairwise_distances(F) # distance metrix for F
    
    gdi = 1-pearsonr(geo_X.flatten(), dist_F.flatten())[0]**2
    
#    from matplotlib import pyplot as plt
#    plt.figure()
#    plt.scatter(geo_X.flatten(), dist_F.flatten())
#    plt.show()
    
#    def cost(alpha):
#        return smape(geo_X*alpha, dist_F)
#    
#    # Cost increases when min(geo_X*alpha) > max(dist_F) or max(geo_X*alpha) < min(dist_F)
#    # But min could be very close to 0 (for points close to each other), so use mean instead
#    bounds=((np.mean(dist_F)/np.max(geo_X), np.max(dist_F)/np.mean(geo_X)),)
#    res = differential_evolution(cost, bounds)
#    
#    gdi = res.fun
    
    return gdi
