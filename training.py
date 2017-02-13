"""
- Gets the xy coordinate representation of sample shapes
- Trains models using PCA, kernel PCA and the autoencoder
- Uses the model to synthesize new shapes

X : shape representation using xy coordinates of the contours
X_l : shape representation after dimensionality reduction of data

Usage: python training.py

Author(s): Wei Chen (wchen459@umd.edu)
"""

import ConfigParser
import math

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.spatial import ConvexHull

from dim_reduction import pca, kpca
from stacked_ae import sae
from ml_ae import mlae
import shape_plot
from parametric_space import initialize
from data_processing import preprocess_features
from metrics import smape, geo_dist_inconsistency, get_geo_dist
from intrinsic_dim import mide
from util import create_dir, reduce_dim, save_model, save_array
import hp_kpca, hp_sae, hp_mlae

    
def train_model(X, X_l, fs, kwargs, intr_dim, dim_F, train, test, c, save_dir, dim_increase, source):
    ''' Build model instances using optimized hyperparameters and evaluate using test data '''
    
    F = np.zeros((X.shape[0], dim_F))
    F_norm = np.zeros_like(F)
    sum_test_errs = []
    sum_test_gdis = []
    geo_X = get_geo_dist(X_l)
    
    for f in fs:
        
        print 'Training ...'
        
        # Get semantic features
        F, name, inv_transform = f(X_l, dim_F, train, test, c=c, **kwargs[f.__name__])

        # Get reconstructed data
        X_rec = dim_increase(inv_transform(F))
        
        create_dir(save_dir + name)
        
        # Preprocess semantic features before plotting
        F_norm, transforms_F = preprocess_features(F)
        # Save the models for transfering features
        save_model(transforms_F[0], name+'_fpca', c)
        save_model(transforms_F[1], name+'_fscaler', c)
        
        # Convex hull of training samples in the semantic space
        if dim_F > 1:
            hull = ConvexHull(F_norm[train])
            boundary = hull.equations
            save_array(boundary, name+'_boundary', c)
        else:
            boundary = None
        
        # Get semantic space sparsity
        kde = KernelDensity(kernel='epanechnikov', bandwidth=0.15).fit(F_norm[train])
        
        print('Saving 2D plots for '+name+' ... ')
        if source == 'glass':
            shape_plot.plot_samples(F_norm, X, X_rec, train, test, save_dir, name, c)
        else:
            shape_plot.plot_samples(F_norm, X, X_rec, train, test, save_dir, name, c, mirror=False)

        if dim_F < 4:
            if source == 'glass':
                shape_plot.plot_grid(7, dim_F, inv_transform, dim_increase, transforms_F, save_dir, 
                                     name, c, boundary, kde)
            else:
                shape_plot.plot_grid(7, dim_F, inv_transform, dim_increase, transforms_F, save_dir, 
                                     name, c, boundary, kde, mirror=False)

        np.savetxt(save_dir+name+'_'+str(c)+'.csv', F, delimiter=",")
        
        # Get reconstruction error
        train_err = smape(X[train], X_rec[train])
        test_err = smape(X[test], X_rec[test])
        print 'Training error: ', train_err
        print 'Testing error: ', test_err
        
        sum_test_errs.append(len(test)*test_err)
        
        # Get topological metrics
        gdi = geo_dist_inconsistency(geo_X, F, X_precomputed=True) # computed for the entire dataset
        print 'GDI: ', gdi
        
        sum_test_gdis.append(X.shape[0]*gdi)
        
    return sum_test_errs, sum_test_gdis


if __name__ == "__main__":
    
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    max_dim_F = config.getint('Global', 'n_features')
    
    X_list, source, sname, n_samples, n_points, noise_scale, source_dir = initialize(verbose=2)
    
    RESULTS_DIR = config.get('Global', 'RESULTS_DIR')
    create_dir(RESULTS_DIR)
    
    example_name = sname + '_%.4f' % noise_scale
    example_dir = RESULTS_DIR + example_name + '/'
    create_dir(example_dir)
    
    test_size = config.getfloat('Global', 'test_size')

    save_dir0 = example_dir + 'n_samples=' + str(n_samples) + '_' + str(test_size) + '/'
    create_dir(save_dir0)
    
    message = 'Source: '+sname+' | Points: '+str(n_points)+' | Samples: '+ \
              str(n_samples)+' | Noise: '+str(noise_scale)+' | Test size: '+str(test_size)
    print message
    
    sum_test_errss = []
    sum_test_gdiss = []
    n_test = 0
    
    # Open the hyperparameter config file
    hp = ConfigParser.ConfigParser()
    hpname = './hp_opt/hp_%s.ini' % example_name
    hp.read(hpname)
    
    c = 0
    for X in X_list:
        
        print '============ Cluster %d ============' % c
        
        n_allc = X.shape[0]
        print 'Sample size: ', n_allc
        if n_allc < 10:
            c += 1
            continue
        
        # Reduce dimensionality
        X_l, dim_increase = reduce_dim(X, plot=False, save=True, c=c)
        
        # Specify training and test set
        n_trainc = int(math.floor(n_allc * (1-test_size)))
        trainc = range(n_trainc)
        testc = range(n_trainc, n_allc)
        n_test += len(testc)
        
        # Estimate intrinsic dimension and nonlinearity
        print 'Estimating intrinsic dimension ...'
        intr_dim = mide(X_l, verbose=1)[0]
        print 'Intrinsic dimension: ', intr_dim
#        nonlinearity = X_l.shape[1] - float(intr_dim)
#        print 'Nonlinearity: ', nonlinearity
        
        if intr_dim < max_dim_F:
            dim_F = intr_dim
        else:
            dim_F = max_dim_F
        
        fs = [
#              pca, 
              kpca,  
#              sae,  
#              mlae,
              ]
        kwargs = {'pca': {}}
        
        # Get optimized hyperparameters
        if kpca in fs:
            section = 'kpca'+str(c)
            if not hp.has_section(section):
                hp_kpca.opt()
                hp.read(hpname)
            kwargs_kpca = {'kernel' : hp.get(section, 'kernel'),
                           'gamma'  : hp.getfloat(section, 'gamma'),\
                           'alpha'  : hp.getfloat(section, 'alpha')}
            kwargs['kpca'] = kwargs_kpca
        
        if sae in fs:
            section = 'stacked_AE'+str(c)
            if not hp.has_section(section):
                hp_sae.opt()
                hp.read(hpname)
            kwargs_sae = {'hidden_size_l1' : hp.getint(section, 'hidden_size_l1'),
                          'hidden_size_l2' : hp.getint(section, 'hidden_size_l2'),\
                          'hidden_size_l3' : hp.getint(section, 'hidden_size_l3'),\
                          'hidden_size_l4' : hp.getint(section, 'hidden_size_l4'),\
                          'p'              : hp.getfloat(section, 'p'),\
                          'l'              : hp.getfloat(section, 'weight_decay'),\
                          'batch_size'     : hp.getint(section, 'batch_size')}
            kwargs['sae'] = kwargs_sae
        
        if mlae in fs:
            section = 'ML_AE'+str(c)
            if not hp.has_section(section):
                hp_mlae.opt()
                hp.read(hpname)
            kwargs_mlae = {'hidden_size_l1' : hp.getint(section, 'hidden_size_l1'),
                           'hidden_size_l2' : hp.getint(section, 'hidden_size_l2'),\
                           'hidden_size_l3' : hp.getint(section, 'hidden_size_l3'),\
                           'hidden_size_l4' : hp.getint(section, 'hidden_size_l4'),\
                           'l'              : hp.getfloat(section, 'weight_decay'),\
                           'lr'             : hp.getfloat(section, 'learning_rate'),\
                           'epsilon'        : hp.getfloat(section, 'epsilon')}
            kwargs['mlae'] = kwargs_mlae
        
        sum_test_errs, sum_test_gdis = train_model(X, X_l, fs, kwargs, intr_dim, dim_F, trainc, testc, 
                                                   c, save_dir0, dim_increase, source)
                                
        sum_test_errss.append(sum_test_errs)
        sum_test_gdiss.append(sum_test_gdis)
        
        c += 1
        
    avg_test_errs = np.sum(np.array(sum_test_errss), axis=0)/n_test
    np.savetxt(save_dir0+'rec_err.csv', avg_test_errs, delimiter=",")
    
    avg_test_gdis = np.sum(np.array(sum_test_gdiss), axis=0)/n_test
    np.savetxt(save_dir0+'topo_gdi.csv', avg_test_gdis, delimiter=",")

    print 'All completed :)'
