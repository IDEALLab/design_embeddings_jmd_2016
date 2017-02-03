"""
Gets the intrinsic dimension of the dataset. 

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
from sklearn.utils.graph import graph_shortest_path
from util import select_neighborhood, find_gap, sort_eigen, get_k_range, get_geo_dist, visualize_graph
from sklearn.neighbors import NearestNeighbors, KernelDensity
from matplotlib import pyplot as plt

    
def he_ding(data, verbose=0):
    ''' Reference:
    "He, J., Ding, L., Jiang, L., Li, Z., & Hu, Q. (2014). Intrinsic dimensionality estimation 
    based on manifold assumption. Journal of Visual Communication and Image Representation, 25(5), 740-747."
    '''
    m = data.shape[0]
    G = get_geo_dist(data)[0] # geodesic distances
    k1 = 5
    k2 = 8
    
    # Compute local intrinsic dimensionality
    G_sorted = np.sort(G)
    local_ids = []
    for i in range(m):
        L1 = G_sorted[i, k1]
        L2 = G_sorted[i, k2]
        local_id = (np.log(k1) - np.log(k2))/(np.log(L1) - np.log(L2))
        local_ids.append(local_id)
    
    # Compute global intrinsic dimensionality
    intr_dim = int(round(np.mean(local_ids), 0))
    
    return intr_dim
    
def lmse(X, nbrs, sigma_n=1, sigma_e=1, verbose=0):
    ''' Local manifold structure estimation '''
    
    m = X.shape[0]
    
    local_dims = []
    for i in range(m):
        X_local = X[nbrs[i]] # local data matrix
        delta = X_local - X[i]
        S = np.diag(1/(sigma_n**2 + sigma_e**2 * np.linalg.norm(delta, axis=1)**2)) # diagonal weight matrix, quad kernel
        XS = np.dot(delta.T, S) # D x m_i
        # Get the Jacobian matrices, i.e., the largest d eigenvectors of matrix XS*XS.T
        w, v = sort_eigen(np.dot(XS,XS.T)) # sorted eigenvalues and eigenvectors     
        if len(w) > 1:
            # Find the local intrinsic dimensionality using eigenvalues
            # This is heuristic
            di = find_gap(w, method='percentage', threshold=.9)+1
        else:
            di = 1
        local_dims.append(di)
    
    return local_dims
    
def mide(X, n_neighbors=None, verbose=0):
    ''' Manifold intrinsic dimension estimator 
    Returns both global intrinsic dimensionality and local intrinsic dimensionality
    '''
    
    # Initial guess
    if n_neighbors is None:
        k_min, k_max = get_k_range(X)
        n_neighbors = (k_min + k_max)/2
    neigh = NearestNeighbors().fit(X)
    dist, nbrs = neigh.kneighbors(n_neighbors=n_neighbors, return_distance=True)
    local_dims = lmse(X, nbrs, verbose=verbose)
    
    if verbose:
        visualize_graph(X, nbrs)
#        plt.figure()
#        plt.plot(local_dims, 'o')
#        plt.title('Local intrinsic dimensions')
#        plt.xlabel('Samples')
#        plt.ylabel('Local ID')
#        plt.ylim(1,4)
#        plt.show()
    
    # Smoothing, this can correct the wrong local dimension estimations
    local_dims = np.array(local_dims)
    X_dims = np.concatenate((X, local_dims.reshape(-1,1)), axis=1)
    b = np.mean(dist[:,-1]) * 5
    kde = KernelDensity(kernel='epanechnikov', bandwidth=b).fit(X_dims)
    for i in range(len(local_dims)):
        Xi = np.concatenate((np.repeat(X[i].reshape(1,-1), len(np.unique(local_dims)), axis=0), 
                             np.unique(local_dims).reshape(-1,1)), axis=1)
        kde_scores = kde.score_samples(Xi)
        local_dims[i] = Xi[np.argmax(kde_scores), -1]
        
    if verbose == 2:
        print local_dims
        
    intr_dim = int(round(np.mean(local_dims), 0))
    
    return intr_dim, local_dims
    