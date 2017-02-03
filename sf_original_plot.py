"""
Plots the superfomula shapes directly using their design parameters.

Author(s): Wei Chen (wchen459@umd.edu)
"""

import os
import ConfigParser
from functools import partial
import numpy as np
from superformula import variables_to_data
import shape_plot

def create_dir(path):
    if os.path.isdir(path): 
        pass 
    else: 
        os.mkdir(path)
    
if __name__ == "__main__":
    
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    
    source = config.get('Global', 'source')
    n_points = config.getint('Global', 'n_points')
    
    orig_space_min = config.getfloat('Superformula', 'orig_space_min')
    orig_space_max = config.getfloat('Superformula', 'orig_space_max')
    alpha = config.getfloat('Superformula', 'nonlinearity')
    beta = config.getint('Superformula', 'n_clusters')
    noise_scale = config.getfloat('Global', 'noise_scale')
    
    results_dir = config.get('Global', 'RESULTS_DIR') + source + '-' + str(beta) + '-' + str(alpha) + '_%.3f/' % noise_scale
    create_dir('results')
    create_dir(results_dir)

    name = 'Superformula (alpha='+str(alpha)+', beta='+str(beta)+')'
    
    # Plot a grid of shapes in original s-t space
    shape_plot.plot_original_grid(8, 2, [[orig_space_min,orig_space_max],[orig_space_min,orig_space_max]], 
                                  partial(variables_to_data, alpha=alpha, beta=beta, n_points=n_points), results_dir, name, mirror=False)
    
    SOURCE_DIR = config.get('Global', 'SOURCE_DIR')
    sname = source + '-' + str(beta) + '-' + str(alpha) + '-' + str(noise_scale)
    orig_space_min = config.getfloat('Superformula', 'orig_space_min')
    orig_space_max = config.getfloat('Superformula', 'orig_space_max')
    source_dir = SOURCE_DIR + 'sf-' + str(orig_space_min) + '-' + str(orig_space_max) + '/'
    variables = np.load(source_dir+'variables.npy')
    
    # Plot samples in original s-t space
    shape_plot.plot_original_samples(8, 2, partial(variables_to_data, alpha=alpha, beta=beta, n_points=n_points), 
                                     results_dir, name, variables, mirror=False)
    
    print name, ": Done generating original space graph!"
    
