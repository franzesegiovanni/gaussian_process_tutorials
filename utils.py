    
import numpy as np
import pyLasaDataset as _PyLasaDataSet
def load_lasa( shape='DoubleBendedLine'):
    DataSet=_PyLasaDataSet.DataSet
    downsample=1
    starting=0
    angle_data = getattr(DataSet, shape)
    demos = angle_data.demos 
    X_tot=(demos[0].pos[:,starting::1])
    Y_tot=np.zeros((2,np.size(X_tot,1)))
    Y_tot[:,:-1]=X_tot[:,1:] - X_tot[:, :-1]
    
    for i in range(1,7):
        X=(demos[i].pos[:,starting::1])
        Y=np.zeros((2,np.size(X,1)))
        Y[:, :-1]=X[:, 1:] - X[:, :-1]
    
        X_tot=np.hstack([X_tot,X])
        Y_tot=np.hstack([Y_tot,Y])
    
    X_tot=np.transpose(X_tot)
    Y_tot=np.transpose(Y_tot)

    X= np.vstack([X_tot[::downsample,:],X_tot[-1,:]])
    Y= np.vstack([Y_tot[::downsample,:],Y_tot[-1,:]])
    
    return X, Y

import itertools
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
import matplotlib as mpl
def plot_GMM(X, Y, means, covariances, index):
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
    splot = plt.subplot(1, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)


     





