"""
Plots reconstruction error vs semantic space dimensionality

Usage: python metric_err_vs_dim.py

Author(s): Wei Chen (wchen459@umd.edu)
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rc("font", size=18)

examples = ['glass', 'sf_linear', 'sf_s_nonlinear', 'sf_v_nonlinear']
titles = {'glass':              'Glass',
          'sf_linear':          'Superformula (linear)',
          'sf_s_nonlinear':     'Superformula (slightly nonlinear)',
          'sf_v_nonlinear':     'Superformula (very nonlinear)'}

n = len(examples)
x = range(1, 6)

for i in range(n):

    plt.figure()
    plt.xticks(np.arange(min(x), max(x)+1, dtype=np.int))
    plt.xlabel('Semantic space dimensionality')
    plt.ylabel('Reconstruction error')
    plt.xlim(0.5, 5.5)
    
    errs = np.zeros((3,5))
    for j in x:
        # Read reconstruction errors in rec_err.txt
        txtfile = open('./results/'+examples[i]+'/n_samples = 115/n_control_points = 20/semantic_dim = '
                       +str(j)+'/rec_err.txt', 'r')
        k = 0
        for line in txtfile:
            errs[k, j-1] = float(line)
            k += 1

    line_pca, = plt.plot(x, errs[0], '-ob', label='PCA')
    line_kpca, = plt.plot(x, errs[1], '-vg', label='Kernel PCA')
    line_ae, = plt.plot(x, errs[2], '-sr', label='Autoencoder')
    plt.legend(handles=[line_pca, line_kpca, line_ae], fontsize=16)
    plt.title(titles[examples[i]])
    fig_name = 'err_vs_dim_'+examples[i]+'.png'
    plt.tight_layout()
    plt.savefig('./results/'+fig_name, dpi=300)
    print fig_name+' saved!'
