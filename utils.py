    
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



     





