"""
Fits and transforms high dimensional data to low dimensional data.

Author(s): Wei Chen (wchen459@umd.edu)
"""

from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.metrics import mean_squared_error
import numpy as np
from util import save_model
    
def pca(data, n_components, train, test, c=None, sample_weight=None, overwrite=True):
    # PCA
    
    data_reduced = np.zeros((data.shape[0],n_components))
    pca = PCA(n_components).fit(data[train])
    data_reduced[train+test] = pca.transform(data[train+test])
    
#    print('explained variance ratio:')
#    print pca.explained_variance_ratio_
    
    name = 'PCA'
    
    if overwrite:
        # Save the model
        save_model(pca, name, c)

    return data_reduced, name, pca.inverse_transform

def kpca(data, n_components, train, test, c=None, sample_weight=None, kernel='linear', 
         gamma=None, degree=3, coef0=1, alpha=0.1, evaluation=False, overwrite=True):
    # Kernel PCA
    
    kpca = KernelPCA(n_components, fit_inverse_transform=True, kernel=kernel, gamma=gamma, degree=degree, 
                     coef0=coef0, alpha=alpha).fit(data[train])

    data_reduced = np.zeros((data.shape[0],n_components))
    data_reduced[train+test] = kpca.transform(data[train+test])
    
    if evaluation:
        data_rec = kpca.inverse_transform(data_reduced[test])
        loss = mean_squared_error(data[test], data_rec)
        return loss
    
    name = 'KPCA'
    
    if overwrite:
        # Save the model
        save_model(kpca, name, c)

    return data_reduced, name, kpca.inverse_transform

def tsvd(data, n_components, train, test, c=None, sample_weight=None, overwrite=True):
    # Truncated SVD
    data_reduced = np.zeros((data.shape[0],n_components))
    tsvd = TruncatedSVD(n_components).fit(data[train])
    data_reduced[train+test] = tsvd.transform(data[train+test])
    
#    print('explained variance ratio:')
#    print(tsvd.explained_variance_ratio_)
        
    name = 'TSVD'
    
    if overwrite:
        # Save the model
        save_model(tsvd, name, c)
        
    return data_reduced, name, tsvd.inverse_transform
