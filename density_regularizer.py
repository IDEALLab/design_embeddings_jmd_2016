"""
Density regularizer for the autoencoder.

Author(s): Wei Chen (wchen459@umd.edu)
"""

import theano.tensor as T
from keras.regularizers import ActivityRegularizer


class DensityRegularizer(ActivityRegularizer):

    def __call__(self, loss):
        output = self.layer.get_output(True)
        dist_sqr = T.sum(T.sqr(output.dimshuffle(0,'x',1)-output.dimshuffle('x',0,1)), axis=2) # Pairwise distance^2
        #dist_sqr = dist_sqr/T.max(dist_sqr) # Scaled by maximum distance^2
        loss += self.l1 * T.mean(T.extra_ops.fill_diagonal(T.inv(dist_sqr + 1e-20), 0)) # Reciprocal
        loss += self.l2 * T.mean(T.extra_ops.fill_diagonal(T.exp(-dist_sqr * 100), 0)) # RBF
        return loss
    
def density_l1(l=.01):
    return DensityRegularizer(l1=l)

def density_l2(l=.01):
    return DensityRegularizer(l2=l)
