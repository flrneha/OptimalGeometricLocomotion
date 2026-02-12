import numpy as np
import pycurve
import scipy.sparse

def innerDissipationPathEnergy(pos, lines, normal, bending_weight,membrane_weight, K, xA=None, xB=None):
    """
    Compute total elastic energy (bending + membrane) across all time steps of the path.
    Energy is summed over consecutive shape pairs and scaled by K (number of time steps) (discrete integration)
    """
    #pos reshape to dof x numVertices x 3
    num_freeShapes = K-1
    if (xA is not None) and (xB is None):
        pos = np.reshape(pos, (num_freeShapes, -1))
        pos = np.concatenate((xA, pos))
    elif xA is None:
        pos = np.reshape(pos, (num_freeShapes, -1))
    else:
        pos = np.reshape(pos, (num_freeShapes, -1)) 
        pos = np.concatenate((xA, pos, xB))
    # Pre-reshape all positions
    pos_reshaped = pos.reshape((pos.shape[0], -1, 3), order='F')
    E  = 0
    for i in range(1,pos.shape[0]):
        E += pycurve.curve_energy(pos_reshaped[i-1], pos_reshaped[i], lines, normal, bending_weight, membrane_weight)
    return K*E

def gradientInnerDissipationPathEnergy(pos, lines, normal, bending_weight, membrane_weight, K, xA=None, xB=None):
    """
    Computes gradient of path energy
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

    gradE  = np.zeros((num_freeShapes, pos.shape[1]))
    # Pre-reshape all positions
    pos_reshaped = pos.reshape((pos.shape[0], -1, 3), order='F')
    if xA is not None:
        for i in range(1,pos.shape[0]-1):
            gradE[i-1] = pycurve.curve_deformed_gradient(pos_reshaped[i-1], pos_reshaped[i], lines, normal, bending_weight, membrane_weight)
            gradE[i-1] += pycurve.curve_undeformed_gradient(pos_reshaped[i], pos_reshaped[i+1], lines, normal, bending_weight, membrane_weight)
   
    if xA is None:
        gradE[0] += pycurve.curve_undeformed_gradient(pos_reshaped[0], pos_reshaped[1], lines, normal, bending_weight, membrane_weight)
        for i in range(1,pos.shape[0]-1):
            gradE[i] = pycurve.curve_deformed_gradient(pos_reshaped[i-1], pos_reshaped[i], lines, normal, bending_weight, membrane_weight)
            gradE[i] += pycurve.curve_undeformed_gradient(pos_reshaped[i], pos_reshaped[i+1], lines, normal, bending_weight, membrane_weight)
    
    if xB is None:
        gradE[-1] = pycurve.curve_deformed_gradient(pos_reshaped[-2], pos_reshaped[-1], lines, normal, bending_weight, membrane_weight)
    return np.reshape(K*gradE, (-1))

def hessianInnerDissipationPathEnergy(pos, lines, normal, bending_weight, membrane_weight, K, xA=None, xB=None):
    """
    Computes Hessian of path energy
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

    num_local_dof = pos.shape[1]
    
    # Pre-reshape all positions
    pos_reshaped = pos.reshape((pos.shape[0], -1, 3), order='F')
    
    # COO format: store triplets (row, col, value)
    estimated_nnz = 4 * num_freeShapes * num_local_dof * num_local_dof  # rough estimate
    rows = np.empty(estimated_nnz, dtype=np.int32)
    cols = np.empty(estimated_nnz, dtype=np.int32)
    data = np.empty(estimated_nnz, dtype=np.float64)
    current_idx = 0  
    def add_block(row_start, col_start, block_matrix):
        nonlocal current_idx
        if scipy.sparse.issparse(block_matrix):
            block_coo = block_matrix.tocoo()
            nnz = block_coo.nnz
            rows[current_idx:current_idx+nnz] = block_coo.row + row_start
            cols[current_idx:current_idx+nnz] = block_coo.col + col_start
            data[current_idx:current_idx+nnz] = block_coo.data
        else:
            nz_mask = block_matrix != 0
            nnz = np.count_nonzero(nz_mask)
            nz_rows, nz_cols = np.nonzero(nz_mask)
            rows[current_idx:current_idx+nnz] = nz_rows + row_start
            cols[current_idx:current_idx+nnz] = nz_cols + col_start
            data[current_idx:current_idx+nnz] = block_matrix[nz_mask]
        current_idx += nnz
    
    if xA is not None:
        add_block(0, 0, pycurve.curve_deformed_hessian(
            pos_reshaped[0], pos_reshaped[1], lines, normal, bending_weight, membrane_weight))
    
    if xA is None:
        # Diagonal blocks
        for k in range(0, num_freeShapes-1):
            add_block((k+1)*num_local_dof, (k+1)*num_local_dof, 
                     pycurve.curve_deformed_hessian(pos_reshaped[k], pos_reshaped[k+1], 
                                                     lines, normal, bending_weight, membrane_weight))
            add_block(k*num_local_dof, k*num_local_dof, 
                     pycurve.curve_undeformed_hessian(pos_reshaped[k], pos_reshaped[k+1], 
                                                       lines, normal, bending_weight, membrane_weight))
        
        # Mixed blocks
        for k in range(0, num_freeShapes-1):
            add_block(k*num_local_dof, (k+1)*num_local_dof, 
                     pycurve.curve_mixed_hessian(pos_reshaped[k], pos_reshaped[k+1], 
                                                  lines, normal, bending_weight, membrane_weight, False))
            add_block((k+1)*num_local_dof, k*num_local_dof, 
                     pycurve.curve_mixed_hessian(pos_reshaped[k], pos_reshaped[k+1], 
                                                  lines, normal, bending_weight, membrane_weight, True))
    
    if xB is not None:
        add_block((num_freeShapes-1)*num_local_dof, (num_freeShapes-1)*num_local_dof, 
                 pycurve.curve_undeformed_hessian(pos_reshaped[-2], pos_reshaped[-1], 
                                                   lines, normal, bending_weight, membrane_weight))
    if xA is not None:
        # Diagonal blocks
        for k in range(0, num_freeShapes-1):
            add_block((k+1)*num_local_dof, (k+1)*num_local_dof, pycurve.curve_deformed_hessian(pos_reshaped[k+1], pos_reshaped[k+2], 
                                                     lines, normal, bending_weight, membrane_weight) )
            add_block(k*num_local_dof, k*num_local_dof, 
                     pycurve.curve_undeformed_hessian(pos_reshaped[k+1], pos_reshaped[k+2], 
                                                       lines, normal, bending_weight, membrane_weight))
        # Mixed blocks
        for k in range(0, num_freeShapes-1):
            add_block(k*num_local_dof, (k+1)*num_local_dof, 
                     pycurve.curve_mixed_hessian(pos_reshaped[k+1], pos_reshaped[k+2], 
                                                  lines, normal, bending_weight, membrane_weight, False))
            add_block((k+1)*num_local_dof, k*num_local_dof, 
                     pycurve.curve_mixed_hessian(pos_reshaped[k+1], pos_reshaped[k+2], 
                                                  lines, normal, bending_weight, membrane_weight, True))

    rows = rows[:current_idx]
    cols = cols[:current_idx]
    data = data[:current_idx]
    # Build sparse matrix from COO format
    Hess = scipy.sparse.coo_matrix((data, (rows, cols)), 
                                    shape=(num_freeShapes * num_local_dof, num_freeShapes * num_local_dof))
    # Convert to CSR for efficient operations
    Hess = Hess.tocsr()
 
    return K * Hess