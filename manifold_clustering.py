"""
Multiple manifolds clustering.

Author(s): Wei Chen (wchen459@umd.edu)
"""

import sys
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import minimize, differential_evolution
import math
from itertools import cycle
from util import find_gap, select_neighborhood, sort_eigen
from intrinsic_dim import mide

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


def align_eigen(theta, X):
    Gs = []
    Vs = []
    c = X.shape[1]
    K = len(theta)
    k = 0
    for i in range(c-1):
        for j in range(i+1, c):
            
            G = np.identity(c) # Givens rotation
            G[i,i] = np.cos(theta[k])
            G[j,j] = np.cos(theta[k])
            G[i,j] = -np.sin(theta[k])
            G[j,i] = np.sin(theta[k])
            Gs.append(G)
            
            V = np.zeros((c,c)) # gradient of G
            V[i,i] = -np.sin(theta[k])
            V[j,j] = -np.sin(theta[k])
            V[i,j] = -np.cos(theta[k])
            V[j,i] = np.cos(theta[k])
            Vs.append(V)
            
            k += 1
            
    Uls = [np.identity(c)]
    for k in range(1, K):
        Ul = np.dot(Uls[k-1], Gs[k-1])
        Uls.append(Ul)
            
    R = np.dot(Uls[K-1], Gs[K-1])
    Z = np.dot(X, R) # alignment result
    
    Urs = [np.identity(c)]
    for k in range(K-2, -1, -1):
        Ur = np.dot(Urs[K-2-k], Gs[k+1])
        Urs.append(Ur)
        Urs.reverse()
        
    return Z, Uls, Vs, Urs

def align_cost(theta, X):
    n = X.shape[0]
    Z = align_eigen(theta, X)[0]
    max_indices = np.argmax(np.abs(Z), axis=1)
    M = Z
    M[range(n), max_indices] = np.zeros(n)
    J = np.sum(M**2)/n # alignment cost
    return J
    
def align_grad(theta, X):
    K = len(theta)
    n = X.shape[0]
    Z, Uls, Vs, Urs = align_eigen(theta, X)
    max_indices = np.argmax(np.abs(Z), axis=1)
    M = Z
    M[range(n), max_indices] = np.zeros(n)
    gradJ = np.zeros(K)
    for k in range(K):
        A = X.dot(Uls[k]).dot(Vs[k]).dot(Urs[k]) # gradient of Z
        B = A
        B[range(n), max_indices] = np.zeros(n) # gradient of M
        gradJ[k] = 2*np.sum(M*B)/n # gradient of alignment cost
    return gradJ
    
def stsc(X, w, C, eigen_gap=0, verbose=0):
    ''' Self-tuning spectral clustering
    affinity_matrix: locally scaled affinity matrix
    C: largest possible group number
    eigen_gap: whether to use gap of eigenvalues to select candinates for STSC
    labels: cluster labels assigned to points
    Reference:
    "Zelnik-Manor, L. and Perona, P. Self-tuning spectral clustering. In NIPS, pp. 1601-1608. 2005."
    '''
    
    if eigen_gap:
        # Get candidates for c_best, based on eigenvalues of L
        c_range = set([idx+1 for idx in find_gap(w[:C+1], multiple=True, verbose=verbose)[0]])
#        if 1 in c_range:
#            c_range = c_range - set({1})
#            c_range.add(2)
    else:
        c_range = range(1, C+1)
    
    min_align = np.inf
    for c in c_range:
        # For each possible group number c recover the rotation which best aligns X's columns 
        # with the canonical coordinate system
        
        if verbose == 2:
            print 'c =', c
        
        if c == 1:
            # In the ideal case, the end points of eigenvectors should form a vertical line in the 1st dim, 
            # i.e., var(X[:, 0]) = 0
            J_opt = np.var(X[:,0])
            theta_opt = 0
        else:
            # Optimize alignment cost over theta
            K = c*(c-1)/2
            
            # Visualize cost function and its gradient
#            if K == 1:
#                thetas = np.arange(-np.pi/2, np.pi/2, np.pi/100)
#                costs = np.zeros_like(thetas)
#                grads = np.zeros_like(thetas)
#                for i in range(len(thetas)):
#                    costs[i] = align_cost(np.array([thetas[i]]), X[:,:2])
#                    grads[i] = align_grad(np.array([thetas[i]]), X[:,:2])
#                plt.figure()
#                plt.subplot(211)
#                plt.plot(thetas, costs)
#                plt.title('cost')
#                plt.subplot(212)
#                plt.plot(thetas, grads)
#                plt.title('gradient')
#                plt.show()
            
            res = differential_evolution(align_cost, bounds=((-np.pi/4, np.pi/4),)*K, 
                                         args=(X[:,:c],), popsize=100, tol=1e-8)
            J_opt = np.clip(res.fun, np.finfo(float).eps, np.inf)
            theta_opt = res.x
        
        if verbose == 2:
            print 'J_opt =', J_opt
            print 'theta_opt =', theta_opt
        
        if J_opt <= min_align:
            min_align = J_opt
            c_best = c
            theta_best = theta_opt
            
    return c_best, theta_best, min_align
    
def get_labels(affinity_matrix, C=5, method='stsc', verbose=0):
    
    D = np.sum(affinity_matrix, axis=1)
    B = np.diag(D**(-.5))
    L = B.dot(affinity_matrix).dot(B) # Ng, Jordan, & Weiss Laplacian
    w, X = sort_eigen(L)
    
    if method == 'stsc':
        # Self-tuning spectural clustering
        c_best, theta_best, min_align = stsc(X, w, C, verbose=verbose)
        
        if c_best == 1:
            labels = np.zeros(affinity_matrix.shape[0], dtype=int)
            Z_best = X[:,:2]
        else:
            Z_best = align_eigen(theta_best, X[:,:c_best])[0]
            labels = np.argmax(Z_best**2, axis=1)
        
        if verbose == 2:
            # Visualize X and Z
            plt.figure()
            plt.subplot(121)
            colorcycler = cycle(colors)
            for c in range(max(labels)+1):
                color = next(colorcycler)
                plt.scatter(X[labels==c,0], X[labels==c,1], s=20, c=color)
            plt.title('X')
            plt.subplot(122)
            colorcycler = cycle(colors)
            for c in range(max(labels)+1):
                color = next(colorcycler)
                plt.scatter(Z_best[labels==c,0], Z_best[labels==c,1], s=20, c=color)
            plt.title('Z')
            plt.show()
            
    else:
        print 'No method called %s!' % method
        sys.exit(0)
    
    if type(labels) is not np.ndarray:
        labels = np.array(labels, dtype=int)
        
    return labels

def rmmsl(X, sigma_n=1, sigma_e=1, sigma_c=.2, verbose=0):
    ''' Robust multiple manifolds structure learning
    X: input data
    radius: radius of nearest neighbors
    sigma_n: the scale of the noise's covariance
    sigma_e: the scale of the error
    Reference:    
    "Gong, D., Zhao, X., and Medioni, G. (2012). Robust multiple manifolds structure 
    learning. arXiv preprint arXiv:1206.4624."
    '''
    m = X.shape[0]
    
    ''' Local manifold structure estimation '''
    if verbose == 2:
        print 'Local manifold structure estimation ...'
        
    local_dims = mide(X, n_neighbors=5, verbose=0)[1]
    nbrs, ind = select_neighborhood(X, local_dims, k_range=(2,20), get_full_ind=True, 
                                    verbose=verbose) # arrays of indices of the nearest points
    K = 7 # used to calculate the local bandwidth
    sigma = np.zeros(m) # local bandwidth
    J = []
    
    for i in range(m):
        X_local = X[nbrs[i]] # local data matrix
        delta = X_local - X[i]
        S = np.diag(1/(sigma_n**2 + sigma_e**2 * np.linalg.norm(delta, axis=1)**2)) # diagonal weight matrix, quad kernel
        XS = np.dot(delta.T, S) # D x m_i
        # Get the Jacobian matrices, i.e., the largest d eigenvectors of matrix XS*XS.T
        w, v = sort_eigen(np.dot(XS,XS.T)) # sorted eigenvalues and eigenvectors
        if len(w) > 1:
            di = find_gap(w, method='percentage')+1 # find the local intrinsic dimensionality using eigenvalues
        else:
            di = 1
        Ji = v[:,:di] # local tangent space, D x d_i
        J.append(Ji)
        # Choose local bandwidth
        # Reference: "Zelnik-Manor, L. and Perona, P. Self-tuning spectral clustering. In NIPS, pp. 1601-1608. 2005."
        sigma[i] = np.linalg.norm(X[i]-X[ind[i][K]])
    
    ''' Glabal manifold structure learning '''
    if verbose == 2:
        print 'Glabal manifold structure learning ...'
        
    # similarity matrix
    W = np.zeros((m,m))
    for i in range(m-1):
        for j in range(i+1, m):
            # Compute principle angles between two tangent spaces using QR factorization and SVD
            # Reference: 
            # "Bjorck, A., & Golub, G. H. (1973). Numerical methods for computing angles 
            # between linear subspaces. Mathematics of computation, 27(123), 579-594."
            Qi, Ri = np.linalg.qr(J[i])
            Qj, Rj = np.linalg.qr(J[j])
            U, s, V = np.linalg.svd(np.dot(Qi.T, Qj))
            s = np.clip(s, 0., 1.)
            theta = np.arccos(s)
            # Compute the similarity matrix
            d_sq = np.inner(X[i]-X[j], X[i]-X[j])
            
            if d_sq == 0:
                W[i,j] = 1
            else:
                # Use modified local bandwidth b, because
                # - it makes point sets with different densities far away (devide their distance by a smaller b)
                # - within a high density cluster, only points that are very close have high simularity
                # - within a low density cluster, points having high simularity is not necessarily close
                b = min(sigma[i], sigma[j])
                w1 = math.exp(-d_sq/b**2) # pairwise distance kernel
                # Use tan(theta): exp(-tan(0)) = 1, exp(-tan(pi/2)) = 0 
                w2 = math.exp(-np.inner(np.tan(theta), np.tan(theta))*b**2/d_sq/sigma_c**2) # curved level kernel
                W[i,j] = w1 * w2
            
    W = W + W.T
    
    return W

def cluster_manifold(X, verbose=0):
    ''' First apply pairwise distance kernel, then use curved level kernel to get subclusters '''
    
    W = rmmsl(X, sigma_c=.2, verbose=verbose) # higher weight on the curved level kernel
    labels = get_labels(W, verbose=verbose)
#    n_clusters = max(labels) + 1
#    n_subc = 1
#    for i in range(n_clusters):
#        # Rearrange labels
#        c = i + n_subc - 1
#        W_sub = rmmsl(X[labels==c], sigma_c=.2, verbose=verbose) # higher weight on the pairwise distance kernel
#        sub_labels = get_labels(W_sub, verbose=verbose)
#        n_subc = max(sub_labels) + 1
#        labels[labels>c] += n_subc-1
#        labels[labels==c] += sub_labels
        
    print 'Number of clusters: ', max(labels)+1

    if verbose:
        # Visualize clustering result
        if X.shape[1] > 3:
            pca = PCA(n_components=3)
            X_plot = pca.fit_transform(X)
        if X.shape[1] < 3:
            X_plot = np.zeros((X.shape[0], 3))
            X_plot[:,:X.shape[1]] = X
        else:
            X_plot = X
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection = '3d')#, aspect='equal')
        colorcycler = cycle(colors)
        for c in range(max(labels)+1):
            color = next(colorcycler)
            ax3d.scatter(X_plot[np.array(labels)==c,0], X_plot[np.array(labels)==c,1], 
                         X_plot[np.array(labels)==c,2], s=20, c=color)
        plt.show()

    return labels
    
    
colors = ['b', 'g', 'y', 'm', 'c', 'r', 'k', 'w']
