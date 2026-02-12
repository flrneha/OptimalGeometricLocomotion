import numpy as np
import utils
import scipy 

'''
Constraining  total_length(positions_next) - total_length(positions)=0 is beneficial over total_length(positions) = L
'''

def total_length( positions ):
    '''
    Computes total arc length along the curve

    Parameters:
    curve_points (array-like): Input curve points with shape (N, 3).
    '''
    L = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))

    return L

def dtotal_length(positions):
    N = len(positions)
    grad = np.zeros(3*N)
    #dL = np.array([ np.array([0.,1.,2.]) + 3*float(i) for i in range(len(dL)) ])
    edge_vec = positions[1:] - positions[:-1]
    edge_norms = np.linalg.norm(edge_vec, axis=1)
    # fill for interior vertices
    for j in range(N-1):
        grad[3*j:3*(j+1)] -= edge_vec[j]/edge_norms[j]
        grad[3*(j+1):3*(j+2) ] += edge_vec[j]/edge_norms[j]

    return grad.reshape((-1, 3)).reshape(-1, order='F')

def length_constraint(positions_next, positions):

    return total_length(positions_next) - total_length(positions)

def path_total_length_constraint(pos, K, xA=None, xB=None):
    num_freeShapes = K-1
    if (xA is not None) and (xB is None):
        pos = np.reshape(pos, (num_freeShapes, -1))
        pos = np.concatenate((xA, pos))
    elif xA is None:
        pos = np.reshape(pos, (num_freeShapes, -1))
    else:
        pos = np.reshape(pos, (num_freeShapes, -1)) 
        pos = np.concatenate((xA, pos, xB))

    return np.array([length_constraint(pos[i+1].reshape((-1,3), order = 'F'), pos[i].reshape((-1,3), order = 'F')) for i in range(pos.shape[0]-1) ]).reshape(-1)

def dpath_total_length_constraint(pos, K, xA=None, xB = None):

    num_freeShapes = K-1
    if (xA is not None) and (xB is None):
        pos = np.reshape(pos, (num_freeShapes, -1))
        pos = np.concatenate((xA, pos))
        freeInd = range(1, num_freeShapes-1)
    elif xA is None:
        pos = np.reshape(pos, (num_freeShapes, -1))
        freeInd = range(num_freeShapes-1)

    else:
        pos = np.reshape(pos, (num_freeShapes, -1)) 
        pos = np.concatenate((xA, pos, xB))
        freeInd = range(1,num_freeShapes-1)

    local_dof =  pos.shape[1]
    
    grad = scipy.sparse.lil_matrix((pos.shape[0]-1,num_freeShapes* pos.shape[1]))

    if xA is not None:
        grad[0, :local_dof] = dtotal_length(pos[1].reshape((-1,3), order = 'F'))

    if xB is not None:
        grad[-1, -local_dof:] = -dtotal_length(pos[-2].reshape((-1,3), order = 'F'))

    j=0

    for k in freeInd:

        grad[k, (j+1)*local_dof:(j+2)*local_dof ] = dtotal_length(pos[k+1].reshape((-1,3), order = 'F'))
        grad[k, j*local_dof:(j+1)*local_dof ] = -dtotal_length(pos[k].reshape((-1,3), order = 'F'))
        j=j+1

    return grad
