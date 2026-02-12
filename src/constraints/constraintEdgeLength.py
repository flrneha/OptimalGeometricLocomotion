import numpy as np
import scipy

"""
Constraint edge lengths (edge_index (list)) of specific shape (index (int))
"""

def create_edge_length_constraint_first_shape(K, bounds = [0,0]):
    """
    Constraint on edge lengths of the first shape in a path with K-1 shapes.
    Total length of the first shape is set to one
    Returns:
        scipy.optimize.NonlinearConstraint
    """
    lb = bounds[0]
    ub = bounds[1]
    scipy_edge_length_constraint = scipy.optimize.NonlinearConstraint(lambda x : edge_length_constraint(x,K),
                                                        jac = lambda x : dedge_length_constraint(x,K),
                                                        lb = lb, ub = ub )
    return scipy_edge_length_constraint

def edge_length_deviation(positions, edge_index, desired_length = None):
    """
        Computes half of the sum of the deviations to desired_length
        default desired lenghth is 1/(N-1)**2, i.e., the edge length of a uniform polygonal curve with N vertices.

        f = \sum .5 (|\gam_k - \gam_{k+1}|^2 - 1/(N-1)**2 )

        returns:
            a len(edge_index) vector
    """
    N = len(positions)
    if desired_length is None:
         desired_length = 1/( N-1 )**2

    if isinstance(edge_index, int):  # Single index case
        edge_length = np.linalg.norm(positions[edge_index + 1] - positions[edge_index]) ** 2
        deviation = edge_length - desired_length
    
    elif isinstance(edge_index, (list, np.ndarray)):     
        edge_index = np.asarray(edge_index)
            
        edge_length = np.linalg.norm( positions[edge_index+1] - positions[edge_index], axis =1)**2
        deviation = edge_length - desired_length


    return deviation

def dedge_length_deviation(positions, edge_index, desired_length = None):
    """
        returns (edge_index) gradients with shape (3n) order as x0,y0,z0, x1,y1,z1, ...
    """
    N = len(positions)
    edge_index = np.atleast_1d(edge_index)
    num_constr = edge_index.size
    grad = np.zeros((num_constr,3*N))
    

    edge_vec = positions[edge_index+1] - positions[edge_index]

    for j in range(num_constr):
        grad[j,3*edge_index[j]:3*(edge_index[j]+1)] -= 2*edge_vec[j]
        grad[j,3*(edge_index[j]+1):3*(edge_index[j]+2) ] += 2*edge_vec[j]

    return grad

def edge_length_constraint(x, K, index = 0, edge_index=None, desired_length = None):
    """
    Set constraint on edge lengths of a specific shape (index) in a path with K-1 shapes.
    input:
        x path with K-1 shapes, index, shape index, edge_index  is a list of edges to constrain, desired_length is the desired edge length.
        edge_index default is None, which means all edges of the shape are constrained
        desired_lenght default is None, which means the edge length of a uniform polygonal curve with N vertices, i.e., 1/(N-1)**2
    returns:
          len(edge_index) vector with the deviation of the edge lengths to the desired length
    """

    x = x.reshape((K-1, -1))
    shape = x[index].reshape((-1,3), order = 'F')

    if edge_index is None:
        edge_index = np.arange((int(shape.shape[0]-1)))
    
    return edge_length_deviation(shape, edge_index, desired_length)

def dedge_length_constraint(x, K, index=0, edge_index=None, desired_length=None):
    """
    input:
        x path with K-1 shapes, index, shape index, edge_index  is a list of edges to constrain, desired_length is the desired edge length.
        edge_index default is None, which means all edges of the shape are constrained
        desired_lenght default is None, which means the edge length of a uniform polygonal curve with N vertices, i.e., 1/(N-1)**2
    returns:
            scipy.sparse.lil_matrix with shape (len(edge_index), (K-1)*local_dof) where local_dof is the number of degrees of freedom of one shape
            
    """
    x = x.reshape((K-1, -1))
    local_dof = x.shape[1]
    N = int(local_dof/3)
    if edge_index is None:
        edge_index = np.arange((N-1))

    edge_index = np.asarray(edge_index)
    num_constr = edge_index.size

    grad_shape = dedge_length_deviation(x[index].reshape((-1,3), order = 'F'), edge_index, desired_length=desired_length)

    grad = scipy.sparse.lil_matrix( ( num_constr, (K-1)* local_dof ) )


    grad[:,index*local_dof:(index +1)*local_dof] = np.array([grad_shape[j].reshape((-1, 3)).flatten(order='F') for j in range(grad_shape.shape[0])])

    return grad 

def edge_length_constraint_all(pos, K, xA = None, xB = None, edge_index= None):
    """
        Constraint for constant edge length of all edge lengths with edge_index of all shapes in pos
    """
    num_freeShapes = K-1
    if (xA is not None) and (xB is None):
        pos = np.reshape(pos, (num_freeShapes, -1))
        pos = np.concatenate((xA, pos))
    elif xA is None:
        pos = np.reshape(pos, (num_freeShapes, -1))
    else:
        pos = np.reshape(pos, (num_freeShapes, -1)) 
        pos = np.concatenate((xA, pos, xB))

    local_dof = pos.shape[1]
    N = int(local_dof/3)
    if edge_index is None:
        edge_index = np.arange((int(N-1)))

    deviation = np.array([edge_length_deviation(pos[i+1].reshape((-1,3), order = 'F'),edge_index=edge_index, desired_length=0)
              -edge_length_deviation(pos[i].reshape((-1,3), order = 'F'), edge_index=edge_index, desired_length=0) for i in range(pos.shape[0]-1) ] ).reshape(-1)
    return deviation

def dedge_length_constraint_all(pos, K, xA=None, xB=None, edge_index=None):

    num_freeShapes = K-1
    if (xA is not None) and (xB is None):
        pos = np.reshape(pos, (num_freeShapes, -1))
        pos = np.concatenate((xA, pos))
        freeInd = range(1, pos.shape[0])
    elif xA is None:
        pos = np.reshape(pos, (num_freeShapes, -1))
        freeInd = range(pos.shape[0])
    else:
        pos = np.reshape(pos, (num_freeShapes, -1)) 
        pos = np.concatenate((xA, pos, xB))
        freeInd = range(1,pos.shape[0]-1)

    local_dof = pos.shape[1]
    N = int(local_dof/3)
    if edge_index is None:
        edge_index = np.arange((int(N-1)))

    edge_index = np.asarray(edge_index)
    num_constr = edge_index.size
    grad_local = np.array([[dedge_length_deviation(pos[i].reshape((-1,3), order = 'F'), edge_index=edge_index, desired_length=0)[j].reshape((-1, 3)).flatten(order='F') for j in range(num_constr)] 
                            for i in freeInd])

    grad = scipy.sparse.lil_matrix((  (pos.shape[0]-1)*num_constr, (K-1)*local_dof ))

    if xA is not None:
        grad[:num_constr, :local_dof] = grad_local[0]
    if xB is not None:
        grad[-num_constr:,-local_dof:] = -grad_local[-1]

    j = 0
    for i in freeInd: #range(1,num_freeShapes-2): # freeInd:
        grad[i*(num_constr):(i+1)*(num_constr), (j+1)*local_dof:(j+2)*local_dof ] = grad_local[j+1]
        grad[i*(num_constr):(i+1)*(num_constr), j*local_dof:(j+1)*local_dof ] = -grad_local[j]
        j = j+1
        if j == len(grad_local)-1:
            break
    return grad







