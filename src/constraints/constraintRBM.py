import numpy as np
import utils
import scipy
"""
constraint for pure rigid body motion between first and last shape
"""

def constraint_rbm(K,rbm):
    """
    Returns the constraint function and its derivatives for enforcing a rigid body motion between the first and last shape in a sequence of K shapes.
    I.e. x[-1] = R*x[0] + b where R,b are given by rbm.

    Args:
        K (int): Number of keyframes.
        rbm (object): Rotation and translation parameters.

    Returns:
        scipy.NonlinearConstraint: Rigid body motion constraint.
    """
    scipy_constraint_rbm = scipy.optimize.NonlinearConstraint(lambda x: rbm_constraint(x, K, rbm),
                                                jac = lambda x: drbm_constraint(x, K, rbm),
                                                lb = -1e-6, ub = 1e-6) # not completely stric is easier to solve
    return scipy_constraint_rbm


def rbm_constraint(x,K, rbm):

    x = x.reshape((K-1, -1))
    A, b = utils.assemble_Rotb(rbm)
    #vertex_weights = utils.compute_vertex_weight(positions)
    start = x[0].reshape((-1,3), order='F')
    end = x[-1].reshape((-1,3), order='F')
    # the squared norm of the individual (3,) vectors in positions of shape (N,3) is np.sum(P ** 2, axis=1)
    diff = end - (utils.dot_mat(A, start) + b) 

    return .5 * np.sum( diff ** 2)

def drbm_constraint(x, K,rbm):
    """
    Computes the gradient of rbm_constraint with respect to x.

    Args:
        x (np.ndarray): Input array of shape (K-1, -1).
        K (int): Number of keyframes.
        rbm (object): Rotation and translation parameters.

    Returns:
        np.ndarray: Gradient of rbm_constraint with respect to x, same shape as x.
    """

    num_freeShapes = K-1
    x = x.reshape((num_freeShapes, -1))
    num_local_dof = x.shape[1]
    A, b = utils.assemble_Rotb(rbm)
    
    start = x[0].reshape((-1, 3), order='F')
    end = x[-1].reshape((-1, 3), order='F')
    
    diff = end - (utils.dot_mat(A, start) + b) 
    
    grad_start = -utils.dot_mat(A.T, diff)
    #grad = scipy.sparse.lil_matrix(( num_freeShapes ,  num_local_dof))
    J = scipy.sparse.lil_matrix(( 1, num_freeShapes * num_local_dof))
    J[0,:num_local_dof] = grad_start.flatten(order='F')
    #grad = np.zeros_like(x)
    #grad[0] = grad_start.flatten(order='F')
    J[0,(K-2)*num_local_dof:] = diff.flatten(order='F')
    #grad[-1] = diff.flatten(order='F')

    return J #grad.flatten()