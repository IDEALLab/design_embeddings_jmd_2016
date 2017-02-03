"""
Synthesizes new shapes using trained models

Author(s): Wei Chen (wchen459@umd.edu)
"""

import ConfigParser
import numpy as np
from util import load_model, load_array, get_fname
from shape_plot import plot_synthesis

def synthesize_shape(attributes, c=0, model_name='KPCA'):
    '''
    attributes : array_like
        Values of shape attributes which have the range [0, 1].
    model_name : str
        Name of the trained model.
    c : int
        The index of a cluster
    X : array_like
        Reconstructed high-dimensional design parameters
    '''
    
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    source = config.get('Global', 'source')
    
    dim = attributes.shape[1]    
    
    transforms = [load_model(model_name+'_fpca', c)]
    transforms.append(load_model(model_name+'_fscaler', c))
    
    model = load_model(model_name, c)
    
    xpca = load_model('xpca', c)
    dim_increase = xpca.inverse_transform
    
    if dim > 1:
        boundary = load_array(model_name+'_boundary', c)
    else:
        boundary = None
    
    # Get save directory
    save_dir = get_fname(model_name, c, directory='./synthesized_shapes/', extension='png')
    
    print('Plotting synthesized shapes for %s_%d ... ' % (model_name, c))
    if dim < 4:            
        if source == 'glass':
            X = plot_synthesis(attributes, model.inverse_transform, dim_increase, transforms, save_dir, 
                               model_name, boundary)
        else:
            X = plot_synthesis(attributes, model.inverse_transform, dim_increase, transforms, save_dir, 
                               model_name, boundary, mirror=False)
                              
    np.save(get_fname(model_name, c, directory='./synthesized_shapes/', extension='npy'), X)
    
                                 
if __name__ == "__main__":
    
    attributes = np.random.rand(1500,3) # Specify shape attributes here
    synthesize_shape(attributes, c=0, model_name='KPCA')
    