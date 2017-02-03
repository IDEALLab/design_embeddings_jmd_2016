"""
Optimizers.

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np

def gradient_descent(f, gradf, x0, args=(), tol=1e-8, maxiter=500, verbose=False):
    
    perturb = 1e-4 * np.random.rand(len(x0))
    L = np.linalg.norm(gradf(x0, *args) - gradf(x0+perturb, *args)) / np.linalg.norm(perturb) # Lipschitz constant
    t = 2/L # initial step size
    y = x0
    x = x0
    delta = 1
#    ress = [np.linalg.norm(gradf(x0, *args))]
    alpha = 0.5
    i = 0
    while i < maxiter and np.linalg.norm(gradf(y, *args)) >= np.linalg.norm(gradf(x0, *args)) * tol:
        x_pre = x
        delta_pre = delta
        x = y - t * gradf(y, *args) # update x
        delta = (1 + np.sqrt(1 + 4 * delta**2)) / 2 # update delta
        
        # Backtracking line search
        while f(x, *args) > f(y, *args) + alpha * np.inner(x-y, gradf(y, *args)):
            t = t/2
            x = y - t * gradf(y, *args) # update x
        
        y = x + (delta_pre - 1) / delta * (x - x_pre) # update y
        res = np.linalg.norm(gradf(y, *args))
#        ress.append(res)
        
        if verbose:
            print 'Iteration %d | Cost: %f | Residual: %f' % (i, f(y, *args), res)
        i += 1
    
    x_sol = y
    
    return x_sol