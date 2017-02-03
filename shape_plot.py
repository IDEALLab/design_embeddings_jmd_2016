"""
Plots samples or new shapes in the semantic space.

Author(s): Wei Chen (wchen459@umd.edu), Jonah Chazan (jchazan@umd.edu)
"""

from matplotlib import pyplot as plt
from sklearn import preprocessing
import numpy as np
import itertools
from data_processing import inverse_features

def plot_shape(xys, attribute_x, attribute_y, ax, mirror, rotate=False, linewidth=1.5, color='blue', alpha=1, scale=.12):
    
    m = xys.reshape(-1,2)
    mx = max([y for (x, y) in m])
    mn = min([y for (x, y) in m])
    xscl = scale / (mx - mn)
    yscl = scale / (mx - mn)
    if rotate:
        m[:,[0,1]] = m[:,[1,0]]
        m[:,1] = -m[:,1]
    ax.plot( *zip(*[(x * xscl + attribute_x, -y * yscl + attribute_y)
                       for (x, y) in m]), linewidth=linewidth, color=color, alpha=alpha)
#    ax.scatter( *zip(*[(x * xscl + attribute_x, -y * yscl + attribute_y)
#                       for (x, y) in m]), s=1, color=color, alpha=alpha)
    if mirror:
        ax.plot( *zip(*[(-x * xscl + attribute_x, -y * yscl + attribute_y) 
                       for (x, y) in m]), linewidth=linewidth, color=color, alpha=alpha)

def plot_samples(features, data, data_rec, train, test, save_path, model_name, cluster, mirror=True):
    
    ''' Create 3D scatter plot and corresponding 2D projections
        of at most the first 3 dimensions of data'''
    
    plt.rc("font", size=font_size)
    n_samples_train = len(train)
    n_samples_test = len(test)
    n_dim = features.shape[1]
    
    if n_dim == 1:
        features = np.concatenate((features, np.zeros_like((features))), axis=1)
        n_dim = 2
    
    if n_dim == 3:
        # Create a 3D scatter plot
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection = '3d')
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([features[:,0].max()-features[:,0].min(), features[:,1].max()-features[:,1].min(), 
                              features[:,2].max()-features[:,2].min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(features[:,0].max()+features[:,0].min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(features[:,1].max()+features[:,1].min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(features[:,2].max()+features[:,2].min())
        ax3d.scatter(Xb, Yb, Zb, c='white', alpha=0)
        ax3d.scatter(features[:,0], features[:,1], features[:,2])
        ax3d.set_title(model_name)
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
        plt.savefig(save_path+model_name+'/'+str(cluster)+'_'+'3d.eps', dpi=600)
        plt.close()
    
    # Project 3D plot to 2D plots and label each point
    figs = []
    ax = []
    k = 0
    for i in range(0, n_dim-1):
        for j in range(i+1, n_dim):
            figs.append(plt.figure())
            ax.append(figs[k].add_subplot(111, aspect='equal'))
            
            # Plot training data
            for index in range(n_samples_train):
                #label = '{0}'.format(index+1)
                #plt.annotate(label, xy = (features[train][index,i], features[train][index,j]), size=10)
                ax[k].scatter(features[train][index,i], features[train][index,j], s = 7)
                plot_shape(data[train][index], features[train][index,i], features[train][index,j], ax[k],
                           mirror, color='red', alpha=.7)
                
                if data_rec is not None:
                    # Draw reconstructed samples for training data
                    plot_shape(data_rec[train][index], features[train][index,i], features[train][index,j], ax[k], 
                               mirror, color='green', alpha=.5)
            
            if len(test) == 0:
                #Plot testing data
                for index in range(n_samples_test):
                    #label = '{0}'.format(index+1)
                    #plt.annotate(label, xy = (features[test][index,i], features[test][index,j]), size=10)
                    ax[k].scatter(features[test][index, i], features[test][index, j], s = 7)
                    plot_shape(data[test][index], features[test][index,i], features[test][index,j], ax[k], 
                               mirror, color='blue', alpha=.7)
                    
                    if data_rec is not None:
                        # Draw reconstructed samples for testing data
                        plot_shape(data_rec[test][index], features[test][index,i], features[test][index,j], ax[k], 
                                   mirror, color='cyan', alpha=.7)            
                
            ax[k].set_title(model_name)
            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.1, 1.1)
            plt.xlabel('Dimension-'+str(i+1))
            plt.ylabel('Dimension-'+str(j+1))
            
            #ax[k].text(-0.1, -0.1, 'training error = '+str(err_train)+' / testing error = '+str(err_test))
            
            k += 1
            plt.tight_layout()
            plt.savefig(save_path+model_name+'/'+str(cluster)+'_'+str(i+1)+'-'+str(j+1)+'.eps', dpi=600)
            plt.close()

def plot_grid(points_per_axis, n_dim, inverse_transform, dim_increase, transforms, save_path, model_name, 
              cluster, boundary=None, kde=None, mirror=True):
    
    ''' Uniformly plots synthesized shape contours in the semantic space.
        If the semantic space is 3D (i.e., n_dim=3), plot one slice of the 3D space at each time. '''
    
    plt.rc("font", size=font_size)
    lincoords = []
    
    for i in range(0,n_dim):
        lincoords.append(np.linspace(0,1,points_per_axis))
    coords_norm = list(itertools.product(*lincoords)) # Create a list of coordinates in the semantic space
    
    coords = inverse_features(coords_norm, transforms) # Min-Max normalization
    if kde is not None:
        # Density evaluation for coords_norm
        kde_scores = np.exp(kde.score_samples(coords_norm))
    else:
        kde_scores = np.ones(len(coords_norm))
    data_rec = dim_increase(inverse_transform(np.array(coords))) # Reconstruct design parameters
    
    # Determine if the i-th item of coords_norm is in the convex hull 
    indices = []
    for i in range(len(coords)):
        c = tuple(coords_norm[i]) + (1,)
        if boundary is not None:
            e = np.dot(boundary, np.expand_dims(c, axis=1))
        if boundary is None or np.all(e <= 0):
            #if kde is None or kde_scores[i] > 0.25:
                indices.append(i)
    
    if n_dim == 1:
        # Create a 1D plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in indices:
            ax.scatter(coords_norm[i][0], 0, s = 7)
            alpha = min(1, kde_scores[i] + .3)
            plot_shape(data_rec[i], coords_norm[i][0], 0, ax, mirror, linewidth=2.0, alpha=alpha)
        
        ax.set_title(model_name, fontsize=20)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 0.1)
        plt.axis('equal')
        plt.xlabel('Dimension-1')
        plt.ylabel('Dimension-2')
        plt.tight_layout()
        plt.savefig(save_path+model_name+'/'+str(cluster)+'_grid.eps', dpi=600)
        
        plt.close()
        
    elif n_dim == 2:
        # Create a 2D plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in indices:
            ax.scatter(coords_norm[i][0], coords_norm[i][1], s = 7)
            alpha = min(1, kde_scores[i] + .3)
            plot_shape(data_rec[i], coords_norm[i][0], coords_norm[i][1], ax, mirror, linewidth=2.0, alpha=alpha)
        
        ax.set_title(model_name, fontsize=20)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('Dimension-1')
        plt.ylabel('Dimension-2')
        plt.tight_layout()
        plt.savefig(save_path+model_name+'/'+str(cluster)+'_grid.eps', dpi=600)
        
#        if kde is not None:
#            for i in indices:
#                # Compute and annotate sparsity for coords_norm[i]
#                #kde_score = np.exp(kde.score_samples(np.reshape(coords_norm[i], (1, -1))))[0]
#                ax.annotate('{:.2f}'.format(kde_scores[i]), (coords_norm[i][0], coords_norm[i][1]), fontsize=12)
#            plt.tight_layout()
#            plt.savefig(save_path+model_name+'/'+str(cluster)+'_grid_sparsity.eps', dpi=600)
        plt.close()
        
    elif n_dim == 3:
        # Create slices of 2D plots for n_dim = 3
        k = 0
        figs = []
        ax = []
        figs.append(plt.figure())
        ax.append(figs[k].add_subplot(111, aspect='equal'))
        xx = coords_norm[indices[0]][0]
        for i in indices:
                        
            if coords_norm[i][0] != xx:
                ax[k].set_title(model_name+' (x = '+str(xx)+')')
                plt.xlim(-0.1, 1.1)
                plt.ylim(-0.1, 1.1)
                plt.xlabel('Dimension-2')
                plt.ylabel('Dimension-3')
                plt.tight_layout()
                plt.savefig(save_path+model_name+'/'+str(cluster)+'_grid_x='+str(xx)+'.eps', dpi=600)
                plt.close()
                
                k += 1
                xx = coords_norm[i][0]
                figs.append(plt.figure())
                ax.append(figs[k].add_subplot(111, aspect='equal'))
                
            ax[k].scatter(coords_norm[i][1], coords_norm[i][2], s = 7)
            alpha = min(1, kde_scores[i] + .3)
            plot_shape(data_rec[i], coords_norm[i][1], coords_norm[i][2], ax[k], mirror, linewidth=2.0, alpha=alpha)
        
        ax[k].set_title(model_name+' (x = '+str(xx)+')')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('Dimension-2')
        plt.ylabel('Dimension-3')
        plt.tight_layout()
        plt.savefig(save_path+model_name+'/'+str(cluster)+'_grid_x='+str(xx)+'.eps', dpi=600)
        plt.close()
        
    else:
        print 'Cannot plot grid for semantic space dimensionality smaller than 1 or larger than 3!'
        
def plot_synthesis(attributes, inverse_transform, dim_increase, transforms, save_path, model_name, 
                   boundary=None, mirror=True):
    
    ''' Given shape attributes, plot synthesized shape contours in given locations of the semantic space. '''
    
    n_dim = attributes.shape[1]    
    plt.rc("font", size=font_size)
    
    raw_attr = inverse_features(attributes, transforms) # Min-Max normalization
    data_rec = dim_increase(inverse_transform(raw_attr)) # Reconstruct design parameters
    
    # Determine if the i-th item of attributes is in the convex hull 
    indices = []
    for i in range(raw_attr.shape[0]):
        c = tuple(attributes[i]) + (1,)
        if boundary is not None:
            e = np.dot(boundary, np.expand_dims(c, axis=1))
        if boundary is None or np.all(e <= 0):
            indices.append(i)
    print '%d valid samples.' % len(indices)
    
    if n_dim == 1:
        # Create a 1D plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in indices:
            ax.scatter(attributes[i,0], 0, s = 7)
            plot_shape(data_rec[i], attributes[i,0], 0, ax, mirror, linewidth=2.0)
        
        ax.set_title(model_name, fontsize=20)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 0.1)
        plt.axis('equal')
        plt.xlabel('Dimension-1')
        plt.ylabel('Dimension-2')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        
        plt.close()
        
    elif n_dim > 1:
        
        if n_dim > 2:
            # use the first two principle attributes
            print 'Warning: the plotted shapes may overlap when the dimension of the attributes is higher than 2!'
            alpha = .7
        else:
            alpha = 1
            
        # Create a 2D plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in indices:
            ax.scatter(attributes[i,0], attributes[i,1], s = 7)
            plot_shape(data_rec[i], attributes[i,0], attributes[i,1], ax, mirror, linewidth=2.0, alpha=alpha)
        
        ax.set_title(model_name, fontsize=20)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('Dimension-1')
        plt.ylabel('Dimension-2')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        
        plt.close()
        
    return data_rec[indices]

def plot_original_samples(points_per_axis, n_dim, inverse_transform, save_path, name,
                          variables, mirror=True):
    
    print "Plotting original samples ..."

    plt.rc("font", size=font_size)
    
    coords = variables
    coords_norm = preprocessing.MinMaxScaler().fit_transform(coords) # Min-Max normalization
    data_rec = inverse_transform(np.array(coords))
    indices = range(len(coords))

    if n_dim == 2:
        # Create a 2D plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in indices:
            ax.scatter(coords_norm[i, 0], coords_norm[i, 1], s = 7)
            plot_shape(data_rec[i], coords_norm[i,0], coords_norm[i,1], ax, mirror, color='red', alpha=.7)

        ax.set_title(name, fontsize=20)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('s')
        plt.ylabel('t')
        plt.tight_layout()
        plt.savefig(save_path+'original_samples.eps', dpi=600)
        
        plt.close()
		
    else:
        print 'Cannot plot original samples for dimensionality other than 2!'
        
def plot_original_grid(points_per_axis, n_dim, min_maxes, inverse_transform, save_path, name, mirror=True):
    
    print "Plotting original grid ..."

    plt.rc("font", size=font_size)
    lincoords = []
    
    for i in range(0,n_dim):
        lincoords.append(np.linspace(min_maxes[i][0],min_maxes[i][1],points_per_axis))
    coords = list(itertools.product(*lincoords)) # Create a list of coordinates in the semantic space
    coords_norm = preprocessing.MinMaxScaler().fit_transform(coords) # Min-Max normalization
    data_rec = inverse_transform(coords)

    indices = range(len(coords))

    if n_dim == 2:
        # Create a 2D plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in indices:
            ax.scatter(coords_norm[i, 0], coords_norm[i, 1], s = 7)
            plot_shape(data_rec[i], coords_norm[i,0], coords_norm[i,1], ax, mirror, linewidth=2)

        ax.set_title(name, fontsize=20)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('s')
        plt.ylabel('t')
        plt.tight_layout()
        plt.savefig(save_path+'original_grid.eps', dpi=600)
        
        plt.close()
        
    else:
        print 'Cannot plot original grid for dimensionality other than 2!'

font_size = 12
linewidth = 2.0