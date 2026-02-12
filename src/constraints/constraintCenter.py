import numpy as np
import scipy.sparse
import scipy

def create_center_constraint(K, center, bounds=[0,0]):
    """
    Returns a scipy constraint object that constrains the center of mass (not weighted with vertex weights) of the last shape to be equal to center.
    Can be used in periodic boundary problem to fix the path in space.
    """
    lb = bounds[0]
    ub = bounds[1]
    scipy_center_constraint =  scipy.optimize.NonlinearConstraint(lambda x : const_center_vec(x, K, center), 
                                                    jac = lambda x : dconst_center_vec(x, K, center),
                                                    hess = lambda x,v : scipy.sparse.lil_matrix((x.size,x.size)) ,
                                                    lb = lb, ub = ub)
    return scipy_center_constraint

def const_center_vec(x, K, center):
    x = x.reshape((K-1, -1))

    x = x[-1,:].reshape((-1,3), order = 'F')
    num_vertex = x.shape[0]
    vertex_weights =  np.ones((num_vertex,1) )

    centroid  = 1./num_vertex * np.sum(vertex_weights * x, axis = 0)
    return centroid - center

def dconst_center_vec(x,K,center):
    num_freeShapes = K-1
    x = x.reshape((num_freeShapes, -1))
    num_local_dof = x.shape[1]

    x = x[-1,:].reshape((-1,3), order = 'F')
    num_vertex = x.shape[0]
    #vertex_weights = np.ones((num_vertex,1 ))
    inde = 1./num_vertex *np.ones(num_vertex)
    J = scipy.sparse.lil_matrix(( 3, num_freeShapes * num_local_dof))
    J[0, (K-2)*num_local_dof:(K-2)*num_local_dof + num_vertex] = inde
    J[1, (K-2)*num_local_dof + num_vertex:(K-2)*num_local_dof + 2*num_vertex] = inde
    J[2, (K-2)*num_local_dof + 2*num_vertex:] = inde
 
    return J

def comp_center(x):
    x = x.reshape((-1,3), order = 'F')
    num_vertex = x.shape[0]
    vertex_weights =  np.ones((num_vertex,1) )

    centroid  = 1./num_vertex * np.sum(vertex_weights * x, axis = 0)
    return centroid