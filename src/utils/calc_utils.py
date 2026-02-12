import numpy as np
import numba

def assemble_Ab(v):
    '''assembles skew-symmetric matrix A from given the first 3 entries of a given (6,) vector v 
       and a translation b from the second 3 entries.
       Input:
         (6,) vector corresponding to an infinitesimal rigid body rotation 
                v[:2] x some_vector + v[3:]
       Returns:
        (3,3) matrix A corresponding to an infinitesimal rotation
        (3,) vector corresponding to the displacement
    '''
    A = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    b = v[3:]

    return A, b
def applyRBM(positions, rbm):
    A, b = assemble_Rotb(rbm)
    return dot_mat(A,positions) + b

def assemble_Rotb(rbm):
    """
    Assembles rotation matrix R from axis-angle representation and translation vector b
    Input:
      (7,) vector corresponding to a rigid body motion [rx, ry, rz, angle, tx, ty, tz]
             rotation around axis v with angle theta + translation b
    """
    v = rbm[:3]
    theta = rbm[3]
    b = rbm[4:]
    v = v / np.linalg.norm(v)  # Normalize the axis vector
    vx, vy, vz = v
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    I = np.eye(3)  # Identity matrix
    A = np.array([
        [0., -vz, vy],
        [vz, 0., -vx],
        [-vy, vx, 0.]
    ])  # Skew-symmetric matrix
    R = cos_theta * I + sin_theta * A + (1 - cos_theta) * np.outer(v, v)
    return R, b

# function which takes in a (N, 3) array of positions and a (3, 3) matrix A and returns an (N,3) array positions of the individual entries P multiplied with A
def dot_mat(A, positions):
    '''Entry-wise matrix-vector product of a matrix (3,3) with an (N,3) array'''
    result = np.dot(A, positions.T)

    return result.T

"""
jit functions
"""
@numba.jit(nopython=True, parallel=False)
def dot_vec(displacements, tangent_vectors):
    """Compute dot product along last axis - JIT version"""
    N = displacements.shape[0]
    result = np.zeros((N, 1))
    for i in range(N):
        result[i, 0] = (displacements[i, 0] * tangent_vectors[i, 0] + 
                        displacements[i, 1] * tangent_vectors[i, 1] + 
                        displacements[i, 2] * tangent_vectors[i, 2])
    return result

@numba.jit(nopython=True)
def manual_swapaxes_last_two(arr):
    """
    Swap the last two axes of a 3D array
    Equivalent to arr.swapaxes(-1, -2) or arr.swapaxes(1, 2)
    """
    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                result[i, j, k] = arr[i, k, j]
    return result

@numba.jit(nopython=True)
def cross_product_3x3_axis0(a, b):
    """
    Manual cross product for (3,3) x (3,3,3) with axis=0
    Returns (3,3,3)
    """
    result = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            result[0, i, j] = a[1, i] * b[2, i, j] - a[2, i] * b[1, i, j]
            result[1, i, j] = a[2, i] * b[0, i, j] - a[0, i] * b[2, i, j]
            result[2, i, j] = a[0, i] * b[1, i, j] - a[1, i] * b[0, i, j]
    return result

@numba.jit(nopython=True)
def cross_product_axis0(a, b):
    """
    Manual cross product equivalent to np.cross(a, b, axis=0)
    where a is (3,) and b is (3, 3, 3)
    Returns (3, 3, 3)
    """
    result = np.zeros_like(b)
    for i in range(b.shape[1]):
        for j in range(b.shape[2]):
            result[0, i, j] = a[1] * b[2, i, j] - a[2] * b[1, i, j]
            result[1, i, j] = a[2] * b[0, i, j] - a[0] * b[2, i, j]
            result[2, i, j] = a[0] * b[1, i, j] - a[1] * b[0, i, j]
    return result

@numba.jit(nopython=True)
def cross_vector_with_3d_array(vec, arr):
    """
    Cross product of a vector (3,) with a 3D array (3, i, j)
    along the first axis, returning (3, i, j)
    Equivalent to np.cross(vec, arr, axis=0) but arr[:,i,j] is the vector being crossed
    """
    result = np.zeros_like(arr)
    for i in range(arr.shape[1]):
        for j in range(arr.shape[2]):
            # Cross product: vec × arr[:, i, j]
            result[0, i, j] = vec[1] * arr[2, i, j] - vec[2] * arr[1, i, j]
            result[1, i, j] = vec[2] * arr[0, i, j] - vec[0] * arr[2, i, j]
            result[2, i, j] = vec[0] * arr[1, i, j] - vec[1] * arr[0, i, j]
    return result

@numba.jit(nopython=True, parallel=False)
def compute_outer_products(tangent_vectors_next, tangent_vectors):
    """Compute outer products for tangent vectors"""
    N = tangent_vectors_next.shape[0]
    T_nc_vec = np.zeros((N, 3, 3))
    T_cc_vec = np.zeros((N, 3, 3))
    
    for k in range(N):
        for i in range(3):
            for j in range(3):
                T_nc_vec[k, i, j] = tangent_vectors_next[k, i] * tangent_vectors_next[k, j]
                T_cc_vec[k, i, j] = tangent_vectors[k, i] * tangent_vectors[k, j]
    
    return T_nc_vec, T_cc_vec

@numba.jit(nopython=True, parallel=False)
def compute_projection_derivatives(tangent_vectors):
    """Compute dT and ddT tensors"""
    N = tangent_vectors.shape[0]
    I = np.eye(3)
    
    dT = np.zeros((N, 3, 3))
    ddT = np.zeros((N, 3, 3, 3))
    
    for k in range(N):
        # dT = I - T ⊗ T
        for i in range(3):
            for j in range(3):
                dT[k, i, j] = I[i, j] - tangent_vectors[k, i] * tangent_vectors[k, j]
        
        # ddT computation
        for i in range(3):
            for l in range(3):
                for m in range(3):
                    ddT[k, i, l, m] = -dT[k, i, l] * tangent_vectors[k, m] - tangent_vectors[k, i] * dT[k, m, l]
    
    return dT, ddT

@numba.jit(nopython=True, parallel=False)
def cross_product_matrix(v, M):
    """Compute cross product v × M along axis 0, returning (3, 3, 3) tensor"""
    result = np.zeros((3, 3, 3))
    
    for i in range(3):
        for j in range(3):
            # v × M[:, i, j]
            result[0, i, j] = v[1] * M[2, i, j] - v[2] * M[1, i, j]
            result[1, i, j] = v[2] * M[0, i, j] - v[0] * M[2, i, j]
            result[2, i, j] = v[0] * M[1, i, j] - v[1] * M[0, i, j]
    
    return result

@numba.jit(nopython=True, parallel=False)
def contract_with_displacement(tensor, midpoint_displacements):
    """
    Compute: T_outer[k, l, m] = -sum_i(ddT[k, i, l, m] * midpoint_displacements[k, i])
    Equivalent to: -np.einsum('kilm, ki -> klm', ddT, midpoint_displacements)
    
    Args:
        ddT: (N-1, 3, 3, 3) tensor
        midpoint_displacements: (N-1, 3) array
    
    Returns:
        T_outer: (N-1, 3, 3) tensor
    """
    N = tensor.shape[0]
    T_outer = np.zeros((N, 3, 3))
    
    for k in range(N):
        for l in range(3):
            for m in range(3):
                sum_val = 0.0
                for i in range(3):
                    sum_val += tensor[k, i, l, m] * midpoint_displacements[k, i]
                T_outer[k, l, m] = sum_val
    
    return T_outer

@numba.jit(nopython=True)
def cross_product_3x3x3(v, M):
    """
    JIT-compiled cross product for v x M where M is (3, 3, 3)
    Returns (3, 3, 3) array where result[:, i, j] = v x M[:, i, j]
    
    Args:
        v: (3,) vector
        M: (3, 3, 3) tensor
    
    Returns:
        result: (3, 3, 3) where result[:, i, j] = v x M[:, i, j]
    """
    result = np.zeros((3, 3, 3))
    
    for i in range(3):
        for j in range(3):
            # v × M[:, i, j]
            result[0, i, j] = v[1] * M[2, i, j] - v[2] * M[1, i, j]
            result[1, i, j] = v[2] * M[0, i, j] - v[0] * M[2, i, j]
            result[2, i, j] = v[0] * M[1, i, j] - v[1] * M[0, i, j]
    
    return result


@numba.jit(nopython=True)
def cross_I_B_axis0(I, B):
    # I: (3, 3)
    # B: (3, 3)
    out = np.empty((3, 3, 3))

    for i in range(3):
        for j in range(3):
            out[0, i, j] = I[1, i] * (B[2, j]) - I[2, i] * ( B[1, j])
            out[1, i, j] = I[2, i] * ( B[0, j]) - I[0, i] * ( B[2, j])
            out[2, i, j] = I[0, i] * ( B[1, j]) - I[1, i] * ( B[0, j])

    return out