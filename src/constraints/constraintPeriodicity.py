import numpy as np
import scipy
import utils

def create_periodic_constraint(K, N, theta, b):
    """
    Returns a scipy NonlinearConstraint that enforces periodic boundary conditions.
    
    Enforces: x_end = R(theta) @ x_start + b
    where R(theta) is a rotation matrix around the z-axis by angle theta.
    
    Args:
        K: Number of time steps (vertices - 1)
        N: Number of spatial points per time step
        theta: Rotation angle (radians) around z-axis
        b: Translation vector (3,) for rigid body motion
        
    Returns:
        scipy.optimize.NonlinearConstraint object
    """
    num_constr = 3*N
    return scipy.optimize.NonlinearConstraint(
        fun=lambda x: compute_periodicity_residual(x, K, theta, b),
        jac=lambda x: compute_periodicity_jacobian(x, K, theta, b),
        hess=lambda x, v: scipy.sparse.csr_matrix((x.size, x.size)),
        lb=np.zeros(num_constr), 
        ub=np.zeros(num_constr)
    )


def compute_periodicity_residual(x, K, theta, b):
    """
    Computes residual for periodic boundary condition: x_end - R(theta) @ x_start - b.
    
    Enforces that the last configuration equals the first configuration after 
    applying rigid body motion (rotation theta around z-axis + translation b).
    
    Args:
        x: Flattened position variables, shape ((K-1)*N*3,)
        K: Number of time steps
        theta: Rotation angle (radians) around z-axis
        b: Translation vector (3,)
        
    Returns:
        Residual vector, shape (3*N,)
    """
    x = x.reshape((K-1, -1))
    
    x_end = x[-1, :].reshape((-1, 3), order='F')
    x_start = x[0, :].reshape((-1, 3), order='F')
    
    # Rotation matrix around z-axis
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R_z = np.array([
        [cos_t, -sin_t, 0.],
        [sin_t,  cos_t, 0.],
        [0.,     0.,    1.]
    ])
    
    return (x_end - utils.dot_mat(R_z, x_start) - b).flatten(order='F')


def compute_periodicity_jacobian(x, K, theta, b):
    """
    Computes Jacobian of periodic boundary condition constraint.
    
    The Jacobian has non-zero blocks only at the first and last time steps:
    - First block: -R(theta) (rotation applied to starting configuration)
    - Last block: I (identity for ending configuration)
    
    Args:
        x: Flattened position variables, shape ((K-1)*N*3,)
        K: Number of time steps
        theta: Rotation angle (radians) around z-axis
        b: Translation vector (3,) - not used in Jacobian
        
    Returns:
        Sparse Jacobian matrix, shape (3*N, (K-1)*3*N)
    """
    x = x.reshape((K-1, -1))
    x_start = x[0, :]
    N = int(len(x_start) / 3)
    
    grad = scipy.sparse.lil_matrix((3*N, (K-1)*3*N))
    
    # Rotation matrix around z-axis
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R_z = np.array([
        [cos_t, -sin_t, 0.],
        [sin_t,  cos_t, 0.],
        [0.,     0.,    1.]
    ])
    
    # Block diagonal matrix of rotations (one per vertex)
    R_z_block = scipy.linalg.block_diag(*([R_z] * N))
    
    # Derivative w.r.t. x_start: -R(theta)
    grad[:len(x_start), :len(x_start)] = -utils.cc_mat_jit(R_z_block)
    
    # Derivative w.r.t. x_end: I
    grad[-len(x_start):, -len(x_start):] = utils.cc_mat_jit(np.identity(len(x_start)))
    
    return grad