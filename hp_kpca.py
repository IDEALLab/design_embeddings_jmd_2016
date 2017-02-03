"""
Hyperparameter optimization for kernel PCA using pysmac

Usage: python hp_kpca.py

Author(s): Wei Chen (wchen459@umd.edu)
"""

import ConfigParser
import pickle
import math
import timeit
import numpy as np
from sklearn.cross_validation import KFold
from dim_reduction import kpca
from intrinsic_dim import mide
from parametric_space import initialize
from util import create_dir, reduce_dim

import pysmac


def cross_validate(gamma, alpha, X, n_folds, n_components):

    # K-fold cross-validation
    kf = KFold(X.shape[0], n_folds=n_folds, shuffle=True)
    i = 1
    loss = 0
    
    for train, test in kf:
        train = train.tolist()
        test = test.tolist()
        
        print 'cross validation: %d' % i
        i += 1
        
        if len(train)>10 and len(test): # if there are enough training and test samples
            # Get cost
            loss += kpca(X, n_components, train, test, kernel='rbf', gamma=gamma, alpha=alpha, evaluation=True)
                                  
        else:
            print 'Please add more samples!'
            
    # Get test reconstruction error
    rec_err_cv = loss/n_folds

    return rec_err_cv

def wrapper(gamma, lg_alpha):
    
    alpha = 10**(lg_alpha)
    
    with open(tempname, 'rb') as f:
        temp = pickle.load(f)
    
    temp[2] += 1 # iteration

    print '----------------------------------------------------'    
    print '%d/%d' % (c+1, len(X_list))
    print 'Iteration:  %d/%d' %(temp[2], n_iter)
    print 'gamma = ', gamma
    print 'alpha = ', alpha
    
    rec_err_cv = cross_validate(gamma, alpha, X_l[trainc], n_folds, n_features)
    
    print 'Result of algorithm run: SUCCESS, %f' % rec_err_cv
    
    if rec_err_cv < temp[0]:
        temp[0:2] = [rec_err_cv, 0]
        temp[3] = [gamma, alpha]
    else:
        temp[1] += 1
			
    with open(tempname, 'wb') as f:
        pickle.dump(temp, f)

    print '********* Optimal configuration **********'
    print 'gamma = ', temp[3][0]
    print 'alpha = ', temp[3][1]

    print 'optimal: ', temp[0]
    print 'count: ', temp[1]
    
    return rec_err_cv


config = ConfigParser.ConfigParser()
config.read('./config.ini')
n_folds = config.getint('Global', 'n_folds')
max_dim = config.getint('Global', 'n_features')

X_list, source, sname, n_samples, n_points, noise_scale, source_dir = initialize()
    
test_size = config.getfloat('Global', 'test_size')

# Open the config file
cfgname = './hp_opt/hp_%s_%.4f.ini' % (sname, noise_scale)
hp = ConfigParser.ConfigParser()
hp.read(cfgname)

start_time = timeit.default_timer()

c = 0
for X in X_list:
    
    if X.shape[0] < 10:
        c += 1
        continue
    
    print '============ Cluster %d ============' % c
    
    # Initialize file to store the reconstruction error and the count
    temp = [np.inf, 0, 0, [0]*2] #[err, count, iteration, optimal parameters]
    create_dir('./hp_opt/temp/')
    tempname = './hp_opt/temp/kpca'
    with open(tempname, 'wb') as f:
        pickle.dump(temp, f)
        
    # Reduce dimensionality
    X_l, dim_increase = reduce_dim(X, plot=False)
    
    n_allc = X.shape[0]
    if n_allc < 10:
        continue
    
    # Specify training and test set
    n_trainc = int(math.floor(n_allc * (1-test_size)))
    print 'Training sample size: ', n_trainc
    trainc = range(n_trainc)
    
    # Estimate intrinsic dimension and nonlinearity
    print 'Estimating intrinsic dimension ...'
    intr_dim = mide(X_l[trainc], verbose=0)[0]
    print 'Intrinsic dimension: ', intr_dim
    
    if intr_dim < max_dim:
        n_features = intr_dim
    else:
        n_features = max_dim
    
    # Define parameters
    parameters=dict(\
                    gamma=('real', [1e-3, 1], 1.0/X_l.shape[1]),
                    lg_alpha=('real', [-10, 0], -5),
                    )

    # Create a SMAC_optimizer object
    opt = pysmac.SMAC_optimizer()
    n_iter = 100

    value, parameters = opt.minimize(wrapper, # the function to be minimized
                                     n_iter, # the number of function calls allowed
                                     parameters) # the parameter dictionary
    
    # Write optimal parameters to the config file
    section = 'kpca'+str(c)
    if not hp.has_section(section):
        # Create the section if it does not exist.
        hp.add_section(section)
        hp.set(section,'kernel','rbf')
        hp.write(open(cfgname,'w'))
        hp.read(cfgname)
    hp.set(section,'gamma',parameters['gamma'])
    hp.set(section,'alpha',10**float(parameters['lg_alpha']))
    hp.write(open(cfgname,'w'))
            
    print(('Lowest function value found: %f'%value))
    print(('Parameter setting %s'%parameters))
    
    c += 1
    
end_time = timeit.default_timer()
training_time = (end_time - start_time)
print 'Training time: %.2f min' % (training_time/60.)
    