"""
Processes parameteric data or semantic features.

Author(s): Wei Chen (wchen459@umd.edu)
"""

from sklearn import preprocessing, decomposition
from manifold_clustering import cluster_manifold
import numpy as np

def preprocess_input(data, center_x=True):
    ''' Centering each sample '''
    if center_x:
        data[:,::2] -= np.mean(data[:,::2], axis=1).reshape(-1, 1)
    data[:,1::2] -= np.mean(data[:,1::2], axis=1).reshape(-1, 1)
    return data

def preprocess_features(features):
    ''' PCA and scaling '''
    pca = decomposition.PCA()
    features = pca.fit_transform(features)
    scaler = preprocessing.MinMaxScaler()
    features_norm = scaler.fit_transform(features)
    transforms = [pca, scaler]
    return features_norm, transforms

def inverse_features(features, transforms):
    transforms.reverse()
    for transform in transforms:
        features = transform.inverse_transform(features)
    return features

def get_indices(labels):
    ''' Get indices for each cluster '''
    if type(labels) is not list:
        labels = labels.tolist()
    cluster_indices = []
    n_clusters = max(labels)+1 # number of clusters
    n_outliers = labels.count(-1) # number of outliers
    n = len(labels) # number of samples
    if n_outliers > .5 * n:
        cluster_indices.append(range(n))
    else:
        for i in range(n_clusters):
            indices = []
            for index, item in enumerate(labels):
                if item == i:
                    indices.append(index)
            cluster_indices.append(indices)
            
#            print 'Cluster ', i+1, ':'
#            print indices
#        print n_outliers, ' outliers'
    
    return cluster_indices

def divide_input(data, verbose=False):
    ''' Manifold clustering '''
    labels = cluster_manifold(data, verbose=verbose)
    cluster_indices = get_indices(labels)
    return cluster_indices
