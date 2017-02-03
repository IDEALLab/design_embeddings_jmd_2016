"""
Combines all the metrics of a method in one plot.

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
from matplotlib import pyplot as plt

plt.rc("font", size=16)
point_size = 40

examples = ['glass', 'sf_linear', 'sf_s_nonlinear', 'sf_v_nonlinear']
titles = {'glass':              'Glass',
          'sf_linear':          'Superformula (linear)',
          'sf_s_nonlinear':     'Superformula (slightly nonlinear)',
          'sf_v_nonlinear':     'Superformula (very nonlinear)'}

n = len(examples)

for i in range(n):

    plt.figure()
    plt.xlabel('Reconstruction error')
    plt.ylabel('Mean ONC')
    #plt.ylim(0, 1)
    
    metrics = np.zeros((2, 3))
    
    # Read reconstruction errors in rec_err.txt
    txtfile1 = open('./results/'+examples[i]+'/n_control_points = 20/semantic_dim = 2/rec_err.txt', 'r')
    l = 0
    for line in txtfile1:
        metrics[0, l] = float(line)
        l += 1
        
    # Read mean ONCs in mean_onc.txt
    txtfile2 = open('./results/'+examples[i]+'/n_control_points = 20/semantic_dim = 2/mean_onc.txt', 'r')
    l = 0
    for line in txtfile2:
        metrics[1, l] = float(line)
        l += 1
    
    plt.scatter(metrics[0, 0], metrics[1, 0], s=point_size)
    plt.annotate('PCA', xy=(metrics[0, 0], metrics[1, 0]), xycoords='data', xytext=(0, 5), textcoords='offset points')
    plt.scatter(metrics[0, 1], metrics[1, 1], s=point_size)
    plt.annotate('Kernel PCA', xy=(metrics[0, 1], metrics[1, 1]), xycoords='data', xytext=(0, 5), textcoords='offset points')
    plt.scatter(metrics[0, 2], metrics[1, 2], s=point_size)
    plt.annotate('Autoencoder', xy=(metrics[0, 2], metrics[1, 2]), xycoords='data', xytext=(-10, 5), textcoords='offset points')
    
    plt.annotate('', xy=(0.2, 0.03), xytext=(0.8, 0.03), xycoords='axes fraction',
                 arrowprops=dict(facecolor='red', shrink=0.1))
    plt.annotate('good', xy=(0.25, 0.06), xycoords='axes fraction', color='r')
    plt.annotate('poor', xy=(0.7, 0.06), xycoords='axes fraction', color='r')
    
    plt.annotate('', xy=(0.03, 0.8), xytext=(0.03, 0.2), xycoords='axes fraction',
                 arrowprops=dict(facecolor='red', shrink=0.1))
    plt.annotate('good', xy=(0.05, 0.7), xycoords='axes fraction', color='r')
    plt.annotate('poor', xy=(0.05, 0.25), xycoords='axes fraction', color='r')
    
    plt.title(titles[examples[i]])
    fig_name = 'metrics_'+examples[i]+'.png'
    plt.tight_layout()
    plt.savefig('./results/'+fig_name, dpi=300)
    print fig_name+' saved!'
    
