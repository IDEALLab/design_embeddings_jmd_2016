"""
Takes in the source of samples (e.g., superformular variables or glassware images),
and gets the xy coordinates of their contours.

data : shape representation using xy coordinates of the contours
data_l : shape representation after dimensionality reduction of data

Usage: python parametric_space.py

Author(s): Wei Chen (wchen459@umd.edu)
"""

import glob
import os
import sys
import ConfigParser
import numpy as np
from superformula import superformula, get_sf_parameters
import random
from data_processing import preprocess_input
from data_processing import divide_input
from util import create_dir, reduce_dim

        
def get_glass_xy(image_paths, n_samples, n_points, n_control_points):
    
    from glass import process_image
    x_plots = []
    for index in range(n_samples):
        print('Processing: ' + os.path.basename(image_paths[index]))
        xy = process_image(image_paths[index], n_control_points, n_points)
        #xy = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1).flatten()
        x_plots.append(xy)
        
    return x_plots

def get_superformula_xy(source_dir, n_samples, n_points):
    
    x_plots = []

    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    
    orig_space_min = config.getfloat('Superformula', 'orig_space_min')
    orig_space_max = config.getfloat('Superformula', 'orig_space_max')
    alpha = config.getfloat('Superformula', 'nonlinearity')
    beta = config.getint('Superformula', 'n_clusters')
    
    fname = source_dir+'variables.npy'
        
    if not os.path.isfile(fname): # If input file .npy not exist in source directory
        variables = source_sf(n_samples, alpha, beta, orig_space_min, orig_space_max)
        create_dir(source_dir)
        np.save(fname, variables)
        print 'Superformula variables saved in %s.' % fname
            
    else:
        print 'Using the existing variables.'
        variables = np.load(fname)
        
    parameters = get_sf_parameters(variables, alpha, beta)
        
    for index in range(n_samples):
        print(str(index+1) + ' - Processing: ' + str(parameters[index]))
        x, y = superformula(*parameters[index], num_points=n_points)
        xy = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1).flatten()
        x_plots.append(xy)

    return x_plots

def source_sf(n_samples, alpha, beta, orig_space_min, orig_space_max):

    variables = []
    for i in range(n_samples):
        s = random.uniform(orig_space_min, orig_space_max)
        t = random.uniform(orig_space_min, orig_space_max)
        variables.append([s, t])

    return np.array(variables)

def add_noise(data, noise_scale=0):
    
    if noise_scale != 0:
        np.random.seed(0)
        scale=np.sqrt(noise_scale/(1.0-noise_scale))
#        data *= np.random.normal(loc=1.0, scale=scale, size=data.shape) # relative
        data += np.random.normal(loc=0.0, scale=np.mean(np.abs(data))*scale, size=data.shape) # absolute
    
    return data

def initialize(verbose=0, raw_data=0):
    
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    
    SOURCE_DIR = config.get('Global', 'SOURCE_DIR')
    source = config.get('Global', 'source')
    n_points = config.getint('Global', 'n_points')
    noise_scale = config.getfloat('Global', 'noise_scale')
    n_samples = config.getint('Global', 'n_samples')
    
    if source == 'sf':
        alpha = config.getfloat('Superformula', 'nonlinearity')
        beta = config.getint('Superformula', 'n_clusters')
        sname = source + '-' + str(beta) + '-' + str(alpha)
        orig_space_min = config.getfloat('Superformula', 'orig_space_min')
        orig_space_max = config.getfloat('Superformula', 'orig_space_max')
        source_dir = SOURCE_DIR + 'sf-' + str(orig_space_min) + '-' + str(orig_space_max) + '/'
        
    elif source == 'glass' or source[:3] == 'sf-':
        sname = source
        source_dir = SOURCE_DIR + source + '/'
        
    else:
        print 'Wrong source!'
        sys.exit(0)
       
    create_dir(source_dir)
    
    # Get parametric data
    fname = source_dir+'raw_parametric_%s.npy' % sname
    if os.path.isfile(fname):
        data = np.load(fname)
        n_samples = min(n_samples, data.shape[0])
        data = data[:n_samples]
        
    else:
        if source == 'glass':
            n_control_points = config.getint('Glass', 'n_control_points')
            image_paths = glob.glob(source_dir+"*.*")
            image_paths.remove(*glob.glob(source_dir+"*.npy"))
            n_samples = min(n_samples, len(image_paths))
            x_plots = get_glass_xy(image_paths, n_samples, n_points, n_control_points)
        elif source == 'sf':
            x_plots = get_superformula_xy(source_dir, n_samples, n_points)
        else:
            print 'No source called %s!' % source
            sys.exit(0)
    
        data = np.zeros((n_samples, 2*n_points))
        for index in range(n_samples):
            data[index,:] = x_plots[index].flatten()
        
        # Shuffle
        np.random.shuffle(data)
        
        # Centering
        if source == 'glass':
            data = preprocess_input(data, center_x=False)
        else:
            data = preprocess_input(data, center_x=True)
            
        np.save(fname, data)
        print 'Parametric data saved in %s.' % fname

    print('Source: '+sname+' | Points: '+str(n_points)+' | Samples: '+str(n_samples)+' | Noise: '+str(noise_scale))
    
    data = add_noise(data, noise_scale) # Add noise
    
    if raw_data:
        return data
    
    data_l, dim_increase = reduce_dim(data, plot=False) # reduce dimensionality
    
    data_list = []
    f0name = SOURCE_DIR+'raw_parametric_%s_%.4f_0.npy' % (sname, noise_scale)
    
    if config.getboolean('Global', 'cluster'):
        
        if not os.path.isfile(f0name):
            # Clustering
            print 'Clustering ...'
            cluster_indices = divide_input(data_l, verbose=verbose)
            
            # Divide .npy file
            c = 0
            for ci in cluster_indices:
                print 'Cluster ', c
                print 'Sample size: ', len(ci)
                fcname = SOURCE_DIR+'raw_parametric_%s_%.4f_%d.npy' % (sname, noise_scale, c)
                np.save(fcname, data[ci])
                print 'Parametric data saved in %s.' % fcname
                data_list.append(data[ci])
                c += 1
                
        else:
            # Directly load data of each cluster from a corresponding .np file
            c = 0
            fcname = f0name
            while os.path.isfile(fcname):
                data_list.append(np.load(fcname))
                if verbose:
                    print 'Cluster ', c
                    print 'Sample size: ', len(data_list[c])
                c += 1
                fcname = SOURCE_DIR+'raw_parametric_%s_%.4f_%d.npy' % (sname, noise_scale, c)
            
    else:
        data_list.append(data)
    
    return data_list, source, sname, n_samples, n_points, noise_scale, source_dir
    
if __name__ == "__main__":
    initialize(verbose=1)
    
    
