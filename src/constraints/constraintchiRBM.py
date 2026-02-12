import numpy as np
import utils
import scipy
import scipy.optimize
import time

def create_chi_constraint(K, rbm, timedict=None, bounds=[0, 0]):
    """
    Returns a scipy constraint object that constrains that the L2 projection of the difference of first and last shape on the space of rigid body motions
    is the rigid body motion specified by rbm.
    For this we set the Euler-Lagrange equations in the rbm parameters of the L2 difference to zero as a constraint.
    """
    lb = bounds[0]
    ub = bounds[1]
    scipy_chi_constraint =  scipy.optimize.NonlinearConstraint(
                                lambda x : drbm_L2_loss(x, K, rbm, timedict=timedict),
                                jac = lambda x : dxdrbm_L2_loss(x, K, rbm, timedict=timedict),
                                lb = lb, ub = ub)
    return scipy_chi_constraint

def rbm_L2_loss(x, K,rbm):
    ''' measures deviation from displacements of start and end shape from being a rigid body motion rbm 

        Args:
            x (np.ndarray): Input array of shape (K-1, -1).
            K (int): Number of keyframes.
            rbm (object): [v1,v2,v3,theta,b1,b2,b3] Rotation axis, rotation angle, translation parameters
        Returns:
            float: L2 difference value.
    '''
    x = x.reshape((K-1, -1))
    A, b = utils.assemble_Rotb(rbm)
    start = x[0].reshape((-1,3), order='F')
    end = x[-1].reshape((-1,3), order='F')
    vertex_weights = utils.compute_vertex_weight(end)

    diff = end - (utils.dot_mat(A, start) + b) 

    return .5 * np.sum(vertex_weights * diff ** 2)

def drbm_L2_loss(x, K, rbm, timedict=None):
    """
    Computes the gradient of L2_diff with respect to rbm parameters.
    
    Args:
        x (np.ndarray): Input array of shape (K-1, -1).
        K (int): Number of keyframes.
        rbm (object): Rotation and translation parameters, Rotation axis, rotation angle, translation parameters
          [v1, v2, v3, theta, b1, b2, b3].
    
    Returns:
        np.ndarray: Gradient of L2_diff with respect to rbm, shape (7,)
    """
    start_time = time.time()

    x = x.reshape((K-1, -1))
    A, b = utils.assemble_Rotb(rbm)
    
    start = x[0].reshape((-1, 3), order='F')
    end = x[-1].reshape((-1, 3), order='F')
    
    vertex_weights = utils.compute_vertex_weight(end)
    # Compute the difference
    diff = end - (utils.dot_mat(A, start) + b)
    weighted_diff = vertex_weights * diff  # (n, 3)
    
    # Extract rotation parameters
    v_unnormalized = rbm[:3]
    v_norm = np.linalg.norm(v_unnormalized)
    v = v_unnormalized / v_norm  # Normalized axis
    vx, vy, vz = v
    
    theta = rbm[3]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Skew-symmetric matrix
    K_matrix = np.array([
        [0., -vz, vy],
        [vz, 0., -vx],
        [-vy, vx, 0.]
    ])
    
    grad_rbm = np.zeros(7)
    
    # ===== Gradient w.r.t. theta =====
    # ∂R/∂theta = -sin(theta)*I + cos(theta)*K + sin(theta)*vv^T
    dR_dtheta = -sin_theta * np.eye(3) + cos_theta * K_matrix + sin_theta * np.outer(v, v)
    dR_dtheta_start = utils.dot_mat(dR_dtheta, start)
    grad_rbm[3] = -np.sum(weighted_diff * dR_dtheta_start)
    
    # ===== Gradient w.r.t. v (axis direction) =====
    # Need to account for normalization: v_hat = v / ||v||
    # Using chain rule: ∂L/∂v_i = ∂L/∂v_hat · ∂v_hat/∂v_i
    
    # First compute ∂L/∂v_hat (gradient w.r.t. normalized v)
    grad_v_normalized = np.zeros(3)
    
    for i in range(3):
        # Derivative of K_matrix w.r.t. v_hat[i]
        dK_dvi = np.zeros((3, 3))
        if i == 0:  # v_x
            dK_dvi[1, 2] = -1.0
            dK_dvi[2, 1] = 1.0
        elif i == 1:  # v_y
            dK_dvi[0, 2] = 1.0
            dK_dvi[2, 0] = -1.0
        else:  # v_z
            dK_dvi[0, 1] = -1.0
            dK_dvi[1, 0] = 1.0
        
        # Derivative of vv^T w.r.t. v_hat[i]
        e_i = np.zeros(3)
        e_i[i] = 1.0
        dvvT_dvi = np.outer(e_i, v) + np.outer(v, e_i)
        
        # ∂R/∂v_hat[i] = sin(theta) * dK/dv_i + (1 - cos(theta)) * d(vv^T)/dv_i
        dR_dvi = sin_theta * dK_dvi + (1 - cos_theta) * dvvT_dvi
        dR_dvi_start = utils.dot_mat(dR_dvi, start)
        grad_v_normalized[i] = -np.sum(weighted_diff * dR_dvi_start)
    
    # Now apply chain rule for normalization: ∂v_hat/∂v = (I - vv^T) / ||v||
    # ∂L/∂v = ∂L/∂v_hat · ∂v_hat/∂v = ∂L/∂v_hat · (I - vv^T) / ||v||
    grad_rbm[:3] = (grad_v_normalized - np.dot(grad_v_normalized, v) * v) / v_norm
    
    # ===== Gradient w.r.t. b (translation) =====
    # ∂L/∂b = -sum(vertex_weights * diff)
    grad_rbm[4:7] = -np.sum(weighted_diff, axis=0)
    
    # Chain rule: ∂L/∂v1, ∂L/∂v2 need correction from v3 dependency
    v1, v2 = rbm[0], rbm[1]
    v3 = rbm[3]
    
    grad_minimal = np.zeros(6)
    grad_minimal[0] = grad_rbm[0] - (v1/v3) * grad_rbm[2]  # ∂v3/∂v1 = -v1/v3
    grad_minimal[1] = grad_rbm[1] - (v2/v3) * grad_rbm[2]  # ∂v3/∂v2 = -v2/v3
    grad_minimal[2:] = grad_rbm[3:]  # theta, b1, b2, b3    
    
    
    if timedict is not None:
        timedict["chiconstraint"] += time.time() - start_time   
    
    return grad_minimal

def dxdrbm_L2_loss(x, K, rbm, timedict=None):
    """
    Computes the Jacobian of dL2_diff_rbm with respect to x.
    Returns a (7, (K-1)*num_local_dof) sparse matrix.
    
    """
    start_time = time.time()
    num_freeShapes = K - 1
    x_reshaped = x.reshape((num_freeShapes, -1))
    num_local_dof = x_reshaped.shape[1]
    n_vertices = num_local_dof // 3
    
    A, b = utils.assemble_Rotb(rbm)
    
    start = x_reshaped[0].reshape((-1, 3), order='F')
    end = x_reshaped[-1].reshape((-1, 3), order='F')
    
    vertex_weights = utils.compute_vertex_weight(end).flatten()
    diff = end - (utils.dot_mat(A, start) + b)
    
    # Rotation parameters
    v_unnormalized = rbm[:3]
    v_norm = np.linalg.norm(v_unnormalized)
    v = v_unnormalized / v_norm
    
    theta = rbm[3]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    K_matrix = np.array([
        [0., -v[2], v[1]],
        [v[2], 0., -v[0]],
        [-v[1], v[0], 0.]
    ])
    
    Jac = scipy.sparse.lil_matrix((7, num_freeShapes * num_local_dof))
    
    # Compute derivative of vertex_weights w.r.t. end
    # dw[i]/d(end[j]) for various i, j
    dw_dend = compute_vertex_weight_gradient(end)  # shape (n, n, 3)
    # ==== Row 3: ∂(∂L/∂theta)/∂x ====
    dR_dtheta = -sin_theta * np.eye(3) + cos_theta * K_matrix + sin_theta * np.outer(v, v)
    dR_dtheta_start = utils.dot_mat(dR_dtheta, start)
    
    grad_theta_start = vertex_weights[:, np.newaxis] * (
        utils.dot_mat(A.T, dR_dtheta_start) - utils.dot_mat(dR_dtheta.T, diff)
    )
    Jac[3, :num_local_dof] = grad_theta_start.flatten(order='F')
    
    # For end: need chain rule for weights
    grad_theta_end = -vertex_weights[:, np.newaxis] * dR_dtheta_start
    
    # Add contribution from derivative of weights
    # grad_rbm[3] = -sum_i w[i] * (diff[i] · dR_dtheta @ start[i])
    # ∂/∂end[j,k] includes ∂w[i]/∂end[j,k] * (-diff[i] · dR_dtheta @ start[i])
    for i in range(n_vertices):
        contribution_i = np.sum(diff[i] * dR_dtheta_start[i])  # scalar
        for j in range(n_vertices):
            for k in range(3):
                grad_theta_end[j, k] -= dw_dend[i, j, k] * contribution_i
    
    Jac[3, (K-2)*num_local_dof:] = grad_theta_end.flatten(order='F')
    
    # ==== Rows 0-2: ∂(∂L/∂v[i])/∂x ====
    P = (np.eye(3) - np.outer(v, v)) / v_norm
    
    for row_idx in range(3):
        grad_v_start = np.zeros_like(start)
        grad_v_end = np.zeros_like(end)
        
        for j in range(3):
            if abs(P[row_idx, j]) < 1e-12:
                continue
            
            dK_dvj = np.zeros((3, 3))
            if j == 0:
                dK_dvj[1, 2], dK_dvj[2, 1] = -1.0, 1.0
            elif j == 1:
                dK_dvj[0, 2], dK_dvj[2, 0] = 1.0, -1.0
            else:
                dK_dvj[0, 1], dK_dvj[1, 0] = -1.0, 1.0
            
            e_j = np.zeros(3)
            e_j[j] = 1.0
            dvvT_dvj = np.outer(e_j, v) + np.outer(v, e_j)
            dR_dvj = sin_theta * dK_dvj + (1 - cos_theta) * dvvT_dvj
            dR_dvj_start = utils.dot_mat(dR_dvj, start)
            
            grad_v_start += P[row_idx, j] * vertex_weights[:, np.newaxis] * (
                utils.dot_mat(A.T, dR_dvj_start) - utils.dot_mat(dR_dvj.T, diff)
            )
            
            grad_v_end_j = P[row_idx, j] * (-vertex_weights[:, np.newaxis] * dR_dvj_start)
            
            # Add contribution from derivative of weights
            for i in range(n_vertices):
                contribution_i = np.sum(diff[i] * dR_dvj_start[i])
                for m in range(n_vertices):
                    for k in range(3):
                        grad_v_end_j[m, k] -= P[row_idx, j] * dw_dend[i, m, k] * contribution_i
            
            grad_v_end += grad_v_end_j
        
        Jac[row_idx, :num_local_dof] = grad_v_start.flatten(order='F')
        Jac[row_idx, (K-2)*num_local_dof:] = grad_v_end.flatten(order='F')
    
    # ==== Rows 4-6: ∂(∂L/∂b[i])/∂x ====
    I = np.identity(3)
    for comp in range(3):
        grad_b_start = vertex_weights[:, np.newaxis] * A[comp, :]
        Jac[4+comp, :num_local_dof] = grad_b_start.flatten(order='F')
        
        grad_b_end = -vertex_weights[:, np.newaxis] * I[comp, :]
        
        # Add contribution from derivative of weights
        # grad_rbm[4+comp] = -sum_i w[i] * diff[i, comp]
        # ∂/∂end[j,k] includes -∂w[i]/∂end[j,k] * diff[i, comp]
        for i in range(n_vertices):
            for j in range(n_vertices):
                for k in range(3):
                    grad_b_end[j, k] -= dw_dend[i, j, k] * diff[i, comp]
        
        Jac[4+comp, (K-2)*num_local_dof:] = grad_b_end.flatten(order='F')
    
    #remove additional degree of freedom
    v1, v2 = rbm[0], rbm[1]
    v3 = rbm[3]
    
    Jac_minimal = scipy.sparse.lil_matrix((6, Jac.shape[1]))
    Jac_minimal[0, :] = Jac[0, :] - (v1/v3) * Jac[2, :]
    Jac_minimal[1, :] = Jac[1, :] - (v2/v3) * Jac[2, :]
    Jac_minimal[2:, :] = Jac[3:, :]
    

    if timedict is not None:
        timedict["Dchiconstraint"] += time.time() - start_time       
    return Jac_minimal


def compute_vertex_weight_gradient(positions):
    """
    Compute the derivative of vertex_weights w.r.t. positions.
    
    Returns:
        dw_dpos: array of shape (n, n, 3) where dw_dpos[i, j, k] = ∂w[i]/∂pos[j, k]
    """
    n = len(positions)
    dw_dpos = np.zeros((n, n, 3))
    
    for i in range(n):
        if i == 0:
            # w[0] = ||pos[1] - pos[0]|| / 2
            edge = positions[1] - positions[0]
            edge_len = np.linalg.norm(edge)
            if edge_len > 1e-12:
                # ∂w[0]/∂pos[0] = -(edge / edge_len) / 2
                dw_dpos[0, 0, :] = -edge / (2 * edge_len)
                # ∂w[0]/∂pos[1] = (edge / edge_len) / 2
                dw_dpos[0, 1, :] = edge / (2 * edge_len)
        
        elif i == n - 1:
            # w[n-1] = ||pos[n-1] - pos[n-2]|| / 2
            edge = positions[i] - positions[i-1]
            edge_len = np.linalg.norm(edge)
            if edge_len > 1e-12:
                # ∂w[n-1]/∂pos[n-2] = -(edge / edge_len) / 2
                dw_dpos[i, i-1, :] = -edge / (2 * edge_len)
                # ∂w[n-1]/∂pos[n-1] = (edge / edge_len) / 2
                dw_dpos[i, i, :] = edge / (2 * edge_len)
        
        else:
            # w[i] = (||pos[i] - pos[i-1]|| + ||pos[i+1] - pos[i]||) / 2
            edge1 = positions[i] - positions[i-1]
            edge1_len = np.linalg.norm(edge1)
            edge2 = positions[i+1] - positions[i]
            edge2_len = np.linalg.norm(edge2)
            
            if edge1_len > 1e-12:
                # ∂(||edge1||)/∂pos[i-1] = -edge1 / edge1_len
                dw_dpos[i, i-1, :] = -edge1 / (2 * edge1_len)
                # ∂(||edge1||)/∂pos[i] = edge1 / edge1_len
                dw_dpos[i, i, :] += edge1 / (2 * edge1_len)
            
            if edge2_len > 1e-12:
                # ∂(||edge2||)/∂pos[i] = -edge2 / edge2_len
                dw_dpos[i, i, :] += -edge2 / (2 * edge2_len)
                # ∂(||edge2||)/∂pos[i+1] = edge2 / edge2_len
                dw_dpos[i, i+1, :] = edge2 / (2 * edge2_len)
    
    return dw_dpos