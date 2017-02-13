"""
Hyperparameter optimization for manifold learning autoencoders using pysmac

Usage: python hp_mlae.py

Author(s): Wei Chen (wchen459@umd.edu)
"""

import ConfigParser
import pickle
import math
import timeit
import numpy as np
from sklearn.cross_validation import KFold
from ml_ae import mlae
from intrinsic_dim import mide
from parametric_space import initialize
from util import create_dir, reduce_dim

import pysmac


def cross_validate(hidden_size_l1, hidden_size_l2, hidden_size_l3, hidden_size_l4, 
                   l, lr, epsilon, X, n_folds, n_components):
                           
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
            loss += mlae(X, n_components, train, test, hidden_size_l1, hidden_size_l2, hidden_size_l3, hidden_size_l4, 
                         l, lr, epsilon, evaluation=True)
                                     
        else:
            print 'Please add more samples!'
            
    # Get test reconstruction error
    rec_err_cv = loss/n_folds

    return rec_err_cv

def wrapper(hidden_size_l1, hidden_size_l2, hidden_size_l3, hidden_size_l4, lg_l, lr, lg_epsilon):
    
    l = 10**lg_l
    epsilon = 10**lg_epsilon
    
    with open(tempname, 'rb') as f:
        temp = pickle.load(f)
    
    temp[2] += 1 # iteration

    print '----------------------------------------------------'    
    print '%d/%d' % (c+1, len(X_list))
    print 'Iteration:  %d/%d' %(temp[2], n_iter)
    print '1st hidden layer size = ', hidden_size_l1
    print '2nd hidden layer size = ', hidden_size_l2
    print '3rd hidden layer size = ', hidden_size_l3
    print '4th hidden layer size = ', hidden_size_l4
    print 'learning rate = ', lr
    print 'epsilon = ', epsilon
    print 'weight decay = ', l
    
    rec_err_cv = cross_validate(hidden_size_l1, hidden_size_l2, hidden_size_l3, hidden_size_l4, 
                                l, lr, epsilon, X_l[trainc], n_folds, n_features)
                               
    print 'Result of algorithm run: SUCCESS, %f' % rec_err_cv
    		
    if rec_err_cv < temp[0]:
        temp[0:2] = [rec_err_cv, 0]
        temp[3] = [hidden_size_l1, hidden_size_l2, hidden_size_l3, hidden_size_l4, lr, epsilon, l]
    else:
        temp[1] += 1
			
    with open(tempname, 'wb') as f:
        pickle.dump(temp, f)

    print '********* Optimal configuration **********'
    print '1st hidden layer size = ', temp[3][0]
    print '2nd hidden layer size = ', temp[3][1]
    print '3rd hidden layer size = ', temp[3][2]
    print '4th hidden layer size = ', temp[3][3]
    print 'learning rate = ', temp[3][4]
    print 'epsilon = ', temp[3][5]
    print 'weight decay = ', temp[3][6]

    print 'optimal: ', temp[0]
    print 'count: ', temp[1]
    
    return rec_err_cv

def opt():
    
    global temp, tempname, n_folds, c, X_list, n_iter, n_features, X_l, trainc
    
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
        # Initialize file to store the reconstruction error and the count
        temp = [np.inf, 0, 0, [0]*2] #[err, count, iteration, optimal parameters]
        create_dir('./hp_opt/temp/')
        tempname = './hp_opt/temp/mlae'
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
        hs_min = n_features+1
        hs_max = int(X_l.shape[1] * 2)
        parameters=dict(
                        hidden_size_l1=('categorical', range(hs_min, hs_max+1)+[0], 0),\
                        hidden_size_l2=('categorical', range(hs_min, hs_max+1)+[0], 0),\
                        hidden_size_l3=('categorical', range(hs_min, hs_max+1)+[0], 0),\
                        hidden_size_l4=('categorical', range(hs_min, hs_max+1)+[0], 0),\
                        lg_l = ('real', [-10, 0.0], -10),\
                        lr = ('real', [1e-4, 2.0], 1.0),\
                        lg_epsilon = ('real', [-10, -4], -8),\
                        )
                        
        # Define constraints                
        forbidden_confs = ["{hidden_size_l1 < hidden_size_l2}",\
                           "{hidden_size_l2 < hidden_size_l3}",\
                           "{hidden_size_l3 < hidden_size_l4}"]
        
        # Create a SMAC_optimizer object
        opt = pysmac.SMAC_optimizer()
        n_iter = 500
        
        value, parameters = opt.minimize(wrapper, # the function to be minimized
                                         n_iter, # the number of function calls allowed
                                         parameters, # the parameter dictionary
                                         forbidden_clauses = forbidden_confs) # constraints
        
        # Write optimal parameters to the config file
        section = 'ML_AE'+str(c)
        if not hp.has_section(section):
            # Create the section if it does not exist.
            hp.add_section(section)
            hp.write(open(cfgname,'w'))
            hp.read(cfgname)
        hp.set(section,'hidden_size_l1',parameters['hidden_size_l1'])
        hp.set(section,'hidden_size_l2',parameters['hidden_size_l2'])
        hp.set(section,'hidden_size_l3',parameters['hidden_size_l3'])
        hp.set(section,'hidden_size_l4',parameters['hidden_size_l4'])
        hp.set(section,'learning_rate',parameters['lr'])
        hp.set(section,'epsilon',10**float(parameters['lg_epsilon']))
        hp.set(section,'weight_decay',10**float(parameters['lg_l']))
        hp.write(open(cfgname,'w'))
                
        print(('Lowest function value found: %f'%value))
        print(('Parameter setting %s'%parameters))
        
        c += 1
        
    end_time = timeit.default_timer()
    training_time = (end_time - start_time)
    print 'Training time: %.2f h' % (training_time/3600.)
    