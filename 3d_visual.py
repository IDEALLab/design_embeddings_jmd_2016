"""
Visualizes design parameters in a 3D space.

Author(s): Wei Chen (wchen459@umd.edu)
"""

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.manifold import Isomap
from util import pick_k
from parametric_space import initialize

X = initialize(raw_data=1)

pca = PCA(n_components=3)
F = pca.fit_transform(X)

# Reconstruction error
X_rec = pca.inverse_transform(F)
err = mean_squared_error(X, X_rec)
print 'Reconstruct error: ', err

#k_opt = pick_k(X, 3)
#F = Isomap(n_neighbors=k_opt, n_components=3).fit_transform(X)

# 3D Plot
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection = '3d')

# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array([F[:,0].max()-F[:,0].min(), F[:,1].max()-F[:,1].min(), F[:,2].max()-F[:,2].min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(F[:,0].max()+F[:,0].min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(F[:,1].max()+F[:,1].min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(F[:,2].max()+F[:,2].min())
ax3d.scatter(Xb, Yb, Zb, c='white', alpha=0)

ax3d.scatter(F[:,0], F[:,1], F[:,2])
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
ax3d.set_xticks([])
ax3d.set_yticks([])
ax3d.set_zticks([])
plt.show()