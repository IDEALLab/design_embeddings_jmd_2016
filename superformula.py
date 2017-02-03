"""
Creates superformula samples using two variables.

Author(s): Jonah Chazan (jchazan@umd.edu), Wei Chen (wchen459@umd.edu)
"""

import numpy as np
import math

def get_sf_parameters(variables, alpha, beta):
    '''
    v[0]: s
    v[1]: t
    '''
    parameters = []
    for v in variables:
        # Set [w, h, m, n1, n2, n3]
        parameters.append([v[0]/10, 1, 3+math.floor(v[0]+v[1])%beta, 1, 8+alpha*(v[0]-10), 8+alpha*(v[1]-10)])
#        parameters.append([v[0]/10, 1, 4+math.floor(v[0]+v[1])%beta, 2, 7+alpha*(v[1]-10), 7+alpha*(v[1]-10)]) # sf-mix
#        parameters.append([v[1]/10, 1, 5+math.floor(v[0]+v[1])%beta, 2, 7+alpha*(v[1]-10), 7+alpha*(v[1]-10)]) # sf-d1
#        parameters.append([v[0]/10, 1, 4+math.floor(v[0]+v[1])%beta, 1, 7+alpha*(v[1]-10), 7+alpha*(v[1]-10)]) # sf-roll
    return  np.array(parameters)

def superformula(w, h, m, n1, n2, n3, num_points=1000):
    phis = np.linspace(0, 2 * math.pi, num_points)

    def r(phi):
        # Force a, b to be 1 so we have a more linear example
        a = 1
        b = 1
        aux = abs(math.cos(m * phi / 4) / a) ** n2 + abs(math.sin(m * phi / 4) / b) ** n3
        return aux ** (-1.0/n1)

    r = np.vectorize(r, otypes=[np.float])

    rs = r(phis)
    
    # Use w, h to scale width and height
    x = w * rs * np.cos(phis)
    y = h * rs * np.sin(phis)
    
    # Scale the shapes so that they have the same height
    mn = min(y)
    mx = max(y)
    h = mx-mn
    y /= h
    x /= h

    return (x, y)
    
def variables_to_data(variables, alpha, beta, n_points):
    parameters = get_sf_parameters(variables, alpha, beta)
    x_plots = []
    for p in parameters:
        x, y = superformula(*p, num_points=n_points)
        xy = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1).flatten()
        x_plots.append(xy)
        
    n_samples = len(variables)
    data = np.zeros((n_samples, 2*n_points))
    for index in range(n_samples):
        data[index,:] = np.reshape(x_plots[index], 2*n_points)
        
    return data
    
