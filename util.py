##########################################
# File: util.py                          #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import re


# raise_if_not_shape
def raise_if_not_shape(name, A, shape):
    """Raise a `ValueError` if the np.ndarray `A` does not have dimensions
    `shape`."""
    if A.shape != shape:
        raise ValueError('{}.shape != {}'.format(name, shape))


# previous_float
PARSE_FLOAT_RE = re.compile(r'([+-]*)0x1\.([\da-f]{13})p(.*)')
def previous_float(x):
    """Return the next closest float (towards zero)."""
    s, f, e = PARSE_FLOAT_RE.match(float(x).hex().lower()).groups()
    f, e = int(f, 16), int(e)
    if f > 0:
        f -= 1
    else:
        f = int('f' * 13, 16)
        e -= 1
    return float.fromhex('{}0x1.{:013x}p{:d}'.format(s, f, e))



##############################################################################
"""
Author(s): Wei Chen (wchen459@umd.edu)
"""

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.utils.graph import graph_shortest_path
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
from sklearn.manifold import Isomap
from sklearn.preprocessing import scale
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from sklearn.externals import joblib
import ConfigParser

def create_dir(path):
    if os.path.isdir(path): 
        pass 
    else: 
        os.mkdir(path)

def reduce_dim(data_h, plot=False, save=False, c=None):
    
    if plot:
        # Scree plot
        plt.rc("font", size=12)
        pca = PCA()
        pca.fit(data_h)
        plt.plot(range(1,data_h.shape[1]+1), pca.explained_variance_ratio_)
        plt.xlabel('Dimensionality')
        plt.ylabel('Explained variance ratio')
        plt.title('Scree Plot')
        plt.show()
        plt.close()
    
    # Dimensionality reduction
    pca = PCA(n_components=.995) # 99.5% variance attained
    data_l = pca.fit_transform(data_h)
    print 'Reduced dimensionality: %d' % data_l.shape[1]
    if save:
        save_model(pca, 'xpca', c)
    
    return data_l, pca.inverse_transform
    
def sort_eigen(M):
    ''' Sort the eigenvalues and eigenvectors in DESCENT order '''
    w, v = np.linalg.eigh(M)
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:,idx]
    return w, v

def find_gap(metrics, threshold=.99, method='difference', multiple=False, verbose=0):
    ''' Find the largest gap of any NONNEGATIVE metrics (which is in DESCENT order)
    The returned index is before the gap
    threshold: needs to be specified only if method is 'percentage'
    multiple: whether to find multiple gaps
    '''
    
    if method == 'percentage':
            
        s = np.sum(metrics)
        for i in range(len(metrics)):
            if np.sum(metrics[:i+1])/s > threshold:
                break
            
        if verbose == 2:
            plt.figure()
            plt.plot(metrics, 'o-')
            plt.title('metrics')
            plt.show()
            
        return i
        
    else:
        if method == 'difference':
            m0 = np.array(metrics[:-1])
            m1 = np.array(metrics[1:])
            d = m0-m1
        elif method == 'divide':
            metrics = np.clip(metrics, np.finfo(float).eps, np.inf)
            m0 = np.array(metrics[:-1])
            m1 = np.array(metrics[1:])
            d = m0/m1
        else:
            print 'No method called %s!' % method
            sys.exit(0)
                
        if multiple:
            
#            dmin = np.min(d)
#            dmax = np.max(d)
#            t = dmin + (dmax-dmin)/10 # set a threshold
#            n_gap = sum(d > t)
#            idx = d.argsort()[::-1][:n_gap]
#            arggap = idx
            
            tol = 1e-4
            arggap = []
            if d[0] > tol:
                arggap.append(0)
            for i in range(len(d)-1):
                if d[i+1] > d[i]:
                    arggap.append(i+1)
            arggap = np.array(arggap)
            
        else:
            arggap = np.argmax(d)
            
        if verbose == 2:
            plt.figure()
            plt.subplot(211)
            plt.plot(metrics, 'o')
            plt.title('metrics')
            plt.subplot(212)
            plt.plot(d, 'o')
#           plt.plot([0, len(d)], [t, t], 'g--')
            plt.title('gaps')
            plt.show()
                
        gap = d[arggap]
        return arggap, gap

def create_graph(X, n_neighbors, include_self=False, verbose=0):
    kng = kneighbors_graph(X, n_neighbors, mode='distance', include_self=include_self)
    nb_graph = graph_shortest_path(kng, directed=False)
        
    if verbose:
        # Visualize nearest neighbor graph
        neigh = NearestNeighbors().fit(X)
        nbrs = neigh.kneighbors(n_neighbors=n_neighbors, return_distance=False)
        visualize_graph(X, nbrs)
    
    return nb_graph
    
def get_geo_dist(X, K='auto', verbose=0):
    
    m = X.shape[0]
    
    if K == 'auto':
        # Choose the smallest k that gives a fully connected graph
        for k in range(2, m):
            G = create_graph(X, k, verbose=verbose)
            if connected_components(G, directed=False, return_labels=False) == 1:
                break;
        return G, k
        
    else:
        return create_graph(X, K, verbose=verbose)
        
def get_k_range(X, verbose=0):
    
    N = X.shape[0]
        
    # Select k_min
    for k in range(1, N):
        G = create_graph(X, k, include_self=False, verbose=verbose)
        if connected_components(G,directed=False,return_labels=False) == 1:
            break;
    k_min = k
    
    # Select k_max
    for k in range(k_min, N):
        kng = kneighbors_graph(X, k, include_self=False).toarray()
        A = np.logical_or(kng, kng.T) # convert to undirrected graph
        P = np.sum(A)/2
        if 2*P/float(N) > k+2:
            break;
    k_max = k-1#min(k_min+10, N)
    
    if verbose == 2:
        print 'k_range: [%d, %d]' % (k_min, k_max)
        
    if k_max < k_min:
        print 'No suitable neighborhood size!'
        
    return k_min, k_max

def get_candidate(X, dim, k_min, k_max, verbose=0):
    errs = []
    k_candidates = []
    for k in range(k_min, k_max+1):
        isomap = Isomap(n_neighbors=k, n_components=dim).fit(X)
        rec_err = isomap.reconstruction_error()
        errs.append(rec_err)
        i = k - k_min
        if i > 1 and errs[i-1] < errs[i-2] and errs[i-1] < errs[i]:
            k_candidates.append(k-1)
            
    if len(k_candidates) == 0:
        k_candidates.append(k)
        
    if verbose == 2:
        print 'k_candidates: ', k_candidates
    
        plt.figure()
        plt.rc("font", size=12)
        plt.plot(range(k_min, k_max+1), errs, '-o')
        plt.xlabel('Neighborhood size')
        plt.ylabel('Reconstruction error')
        plt.title('Select candidates of neighborhood size')
        plt.show()
        
    return k_candidates

def pick_k(X, dim, k_min=None, k_max=None, verbose=0):
    ''' Pick optimal neighborhood size for isomap algothm
    Reference:
    Samko, O., Marshall, A. D., & Rosin, P. L. (2006). Selection of the optimal parameter 
    value for the Isomap algorithm. Pattern Recognition Letters, 27(9), 968-979.
    '''
    
    if k_min is None or k_max is None:
        k_min, k_max = get_k_range(X, verbose=verbose)
    
    ccs = []
    k_candidates = range(k_min, k_max+1)#get_candidate(X, dim, k_min, k_max, verbose=verbose)
    for k in k_candidates:
        isomap = Isomap(n_neighbors=k, n_components=dim).fit(X)
        F = isomap.fit_transform(X)
        distF = pairwise_distances(F)
        distX = create_graph(X, k, verbose=verbose)
        cc = 1-pearsonr(distX.flatten(), distF.flatten())[0]**2
        ccs.append(cc)
       
    k_opt = k_candidates[np.argmin(ccs)]
    
    if verbose == 2:
        print 'k_opt: ', k_opt
        
        plt.figure()
        plt.rc("font", size=12)
        plt.plot(k_candidates, ccs, '-o')
        plt.xlabel('Neighborhood size')
        plt.ylabel('Residual variance')
        plt.title('Select optimal neighborhood size')
        plt.show()
    
    return k_opt
    
def estimate_dim(data, verbose=0):
    ''' Estimate intrinsic dimensionality of data
    data: input data
    Reference:
    "Samko, O., Marshall, A. D., & Rosin, P. L. (2006). Selection of the optimal parameter 
    value for the Isomap algorithm. Pattern Recognition Letters, 27(9), 968-979."
    '''
    # Standardize by center to the mean and component wise scale to unit variance
    data = scale(data)
    # The reconstruction error will decrease as n_components is increased until n_components == intr_dim
    errs = []
    found = False
    k_min, k_max = get_k_range(data, verbose=verbose)
    for dim in range(1, data.shape[1]+1):
        k_opt = pick_k(data, dim, k_min, k_max, verbose=verbose)  
        isomap = Isomap(n_neighbors=k_opt, n_components=dim).fit(data)
        err = isomap.reconstruction_error()
        #print(err)
        errs.append(err)
        
        if dim > 2 and errs[dim-2]-errs[dim-1] < .5 * (errs[dim-3]-errs[dim-2]):
                intr_dim = dim-1
                found = True
                break
        
    if not found:
        intr_dim = 1
        
#        intr_dim = find_gap(errs, method='difference', verbose=verbose)[0] + 1
#        intr_dim = find_gap(errs, method='percentage', threshold=.9, verbose=verbose) + 1

    if verbose == 2:
        plt.figure()
        plt.rc("font", size=12)
        plt.plot(range(1,dim+1), errs, '-o')
        plt.xlabel('Dimensionality')
        plt.ylabel('Reconstruction error')
        plt.title('Select intrinsic dimension')
        plt.show()
    
    return intr_dim
   
def get_singular_ratio(X_nbr, d):
    x_mean = np.mean(X_nbr, axis=1).reshape(-1,1)
    s = np.linalg.svd(X_nbr-x_mean, compute_uv=0)
    r = (np.sum(s[d:]**2.)/np.sum(s[:d]**2.))**.5
    return r
    
def select_neighborhood(X, dims, k_range=None, get_full_ind=False, verbose=0):
    ''' Inspired by the Neighborhood Contraction and Neighborhood Expansion algorithms
    The selected neighbors for each sample point should reflect the local geometric structure of the manifold
    Reference:
    "Zhang, Z., Wang, J., & Zha, H. (2012). Adaptive manifold learning. IEEE Transactions 
    on Pattern Analysis and Machine Intelligence, 34(2), 253-265."
    '''
    print 'Selecting neighborhood ... '
    m = X.shape[0]
    
    if type(dims) == int:
        dims = [dims] * m
    
    if k_range is None:
        k_min, k_max = get_k_range(X)
    else:
        k_min, k_max = k_range
    
#    G = get_geo_dist(X, verbose=verbose)[0] # geodesic distances
#    ind = np.argsort(G)[:,:k_max+1]
    neigh = NearestNeighbors().fit(X)
    ind = neigh.kneighbors(n_neighbors=k_max, return_distance=False)
    ind = np.concatenate((np.arange(m).reshape(-1,1), ind), axis=1)
    nbrs = []
                
    # Choose eta
    k0 = k_max
    r0s =[]
    for j in range(m):
        X_nbr0 = X[ind[j,:k0]].T
        r0 = get_singular_ratio(X_nbr0, dims[j])
        r0s.append(r0)
    r0s.sort(reverse=True)
    j0 = find_gap(r0s, method='divide')[0]
    eta = (r0s[j0]+r0s[j0+1])/2
#    eta = 0.02
    if verbose:
        print 'eta = %f' % eta
    
    for i in range(m):
        
        ''' Neighborhood Contraction '''
        rs = []
        for k in range(k_max, k_min-1, -1):        
            X_nbr = X[ind[i,:k]].T
            r = get_singular_ratio(X_nbr, dims[i])
            rs.append(r)
            if r < eta:
                ki = k
                break
        if k == k_min:
            ki = k_max-np.argmin(rs)
        nbrs.append(ind[i,:ki])
        
        ''' Neighborhood Expansion '''
        pca = PCA(n_components=dims[i]).fit(X[nbrs[i]])
        nbr_out = ind[i, ki:] # neighbors of x_i outside the neighborhood set by Neighborhood Contraction
        for j in nbr_out:
            theta = pca.transform(X[j].reshape(1,-1))
            err = np.linalg.norm(pca.inverse_transform(theta) - X[j]) # reconstruction error
            if err < eta * np.linalg.norm(theta):
                nbrs[i] = np.append(nbrs[i], [j])

#        print ki, len(nbrs[i])
#    print max([len(nbrs[i]) for i in range(m)])
        
    if verbose:
        # Visualize nearest neighbor graph
        visualize_graph(X, nbrs)
        
        # Visualize neighborhood selection
        if X.shape[1] > 3:
            pca = PCA(n_components=3)
            F = pca.fit_transform(X)
        else:
            F = np.zeros((X.shape[0], 3))
            F[:,:X.shape[1]] = X
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection = '3d')
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([F[:,0].max()-F[:,0].min(), F[:,1].max()-F[:,1].min(), F[:,2].max()-F[:,2].min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(F[:,0].max()+F[:,0].min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(F[:,1].max()+F[:,1].min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(F[:,2].max()+F[:,2].min())
        ax3d.scatter(Xb, Yb, Zb, c='white', alpha=0)
        # Plot point sets in 3D
        plot_samples = [0, 1]
        nbr_indices = []
        for i in plot_samples:
            nbr_indices = list(set(nbr_indices) | set(nbrs[i]))
        F_ = np.delete(F, nbr_indices, axis=0)
        ax3d.scatter(F_[:,0], F_[:,1], F_[:,2], c='white')
        colors = ['b', 'g', 'y', 'r', 'c', 'm', 'y', 'k']
        from itertools import cycle
        colorcycler = cycle(colors)
        for i in plot_samples:
            color = next(colorcycler)
            ax3d.scatter(F[nbrs[i][1:],0], F[nbrs[i][1:],1], F[nbrs[i][1:],2], marker='*', c=color, s=100)
            ax3d.scatter(F[i,0], F[i,1], F[i,2], marker='x', c=color, s=100)
        plt.show()
        
    if get_full_ind:
        return nbrs, ind
    else:
        return nbrs
        
def visualize_graph(X, nbrs):
    
    # Reduce dimensionality
    if X.shape[1] > 3:
        pca = PCA(n_components=3)
        F = pca.fit_transform(X)
    else:
        F = np.zeros((X.shape[0], 3))
        F[:,:X.shape[1]] = X
        
    m = F.shape[0]
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection = '3d')
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([F[:,0].max()-F[:,0].min(), F[:,1].max()-F[:,1].min(), F[:,2].max()-F[:,2].min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(F[:,0].max()+F[:,0].min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(F[:,1].max()+F[:,1].min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(F[:,2].max()+F[:,2].min())
    ax3d.scatter(Xb, Yb, Zb, c='white', alpha=0)
    # Plot point sets in 3D
    ax3d.scatter(F[:,0], F[:,1], F[:,2], c='blue')
    # Plot edges
#    for i in range(m-1):
#        for j in range(i+1, m):
#            if j in nbrs[i]:
#                line = np.vstack((F[i], F[j]))
#                ax3d.plot(line[:,0], line[:,1], line[:,2], c='green')
    for i in [3]:
        for j in range(i+1, m):
            if j in nbrs[i]:
                line = np.vstack((F[i], F[j]))
                ax3d.plot(line[:,0], line[:,1], line[:,2], c='green')
    plt.show()

def get_fname(mname, c, directory='./trained_models/', extension='pkl'):
    
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    source = config.get('Global', 'source')
    noise_scale = config.getfloat('Global', 'noise_scale')
    
    if source == 'sf':
        alpha = config.getfloat('Superformula', 'nonlinearity')
        beta = config.getint('Superformula', 'n_clusters')
        sname = source + '-' + str(beta) + '-' + str(alpha)
        
    elif source == 'glass' or source[:3] == 'sf-':
        sname = source
    
    if c is None:
        fname = '%s/%s_%.4f_%s.%s' % (directory, sname, noise_scale, mname, extension)
    else:
        fname = '%s/%s_%.4f_%s_%d.%s' % (directory, sname, noise_scale, mname, c, extension)
    
    return fname
    
def save_model(model, mname, c=None):
    
    # Get the file name
    fname = get_fname(mname, c)
    # Save the model
    joblib.dump(model, fname, compress=9)
    print 'Model ' + mname + ' saved!'
    
def load_model(mname, c=None):
    
    # Get the file name
    fname = get_fname(mname, c)
    # Load the model
    model = joblib.load(fname)
    return model
    
def save_array(array, dname, c=None):
    
    # Get the file name
    fname = get_fname(dname, c, extension='npy')
    # Save the model
    np.save(fname, array)
    print 'Model ' + dname + ' saved!'
    
def load_array(dname, c=None):
    
    # Get the file name
    fname = get_fname(dname, c, extension='npy')
    # Load the model
    array = np.load(fname)
    return array
