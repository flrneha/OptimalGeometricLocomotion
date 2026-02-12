import numpy as np
import scipy.sparse
from .outer_dissipation import *

def outerDissipationPathEnergy(pos,K, xA=None, xB=None,a_weight = 1., b_weight=-0.95, mass_factor=None,timedict = None):
    """
    Computes path energy with respect to outer dissipation energy
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
    E  = 0
    pos_reshaped = pos.reshape((pos.shape[0], -1, 3), order='F')

    for i in range(1,pos.shape[0]):
        displacements =  pos_reshaped[i] - pos_reshaped[i-1]
        E += compute_external_edge_dissipation(displacements, pos_reshaped[i-1], 
                                               a_weight =a_weight, b_weight = b_weight, mass_factor=mass_factor)
    return K*E

def gradientOuterDissipationPathEnergy(pos, K, xA=None, xB=None,a_weight = 1., b_weight=-0.95, mass_factor=None, timedict = None):
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
    pos_reshaped = pos.reshape((pos.shape[0], -1, 3), order='F')

    if xA is not None:
        for i in range(1,pos.shape[0]-1):
            displacements_i =  pos_reshaped[i] - pos_reshaped[i-1]
            displacements_iplus =  pos_reshaped[i+1] - pos_reshaped[i]
            
            gradE[i-1] += np.reshape(compute_gradient_external_edge_dissipation(-displacements_i, pos_reshaped[i], a_weight =a_weight, 
                                                                                b_weight = b_weight,mass_factor=mass_factor), -1, order = 'F')
            gradE[i-1] += np.reshape(compute_gradient_external_edge_dissipation(displacements_iplus, pos_reshaped[i],  a_weight =a_weight, 
                                                                                b_weight = b_weight,mass_factor=mass_factor), -1, order = 'F')

    if xA is None:
            
        displacements_iplus =  pos_reshaped[1] - pos_reshaped[0]

        gradE[0] += np.reshape(compute_gradient_external_edge_dissipation(displacements_iplus, pos_reshaped[0],  a_weight =a_weight, 
                                                                          b_weight = b_weight,mass_factor=mass_factor), -1, order = 'F')
        for i in range(1,pos.shape[0]-1):
            displacements_i =  pos_reshaped[i] - pos_reshaped[i-1]
            displacements_iplus =  pos_reshaped[i+1] - pos_reshaped[i]
            
            #gradient wrt positions next
            gradE[i] += np.reshape(compute_gradient_external_edge_dissipation(-displacements_i, pos_reshaped[i], a_weight =a_weight, 
                                                                              b_weight = b_weight,mass_factor=mass_factor), -1, order = 'F')
            #gradient wrt positions
            gradE[i] += np.reshape(compute_gradient_external_edge_dissipation(displacements_iplus, pos_reshaped[i],  a_weight =a_weight, 
                                                                              b_weight = b_weight,mass_factor=mass_factor), -1, order = 'F')
      

    if xB is None:
        gradE[-1] = np.reshape(compute_gradient_external_edge_dissipation(-(pos_reshaped[-1] - pos_reshaped[-2]), pos_reshaped[-1],  a_weight =a_weight, 
                                                                          b_weight = b_weight,mass_factor=mass_factor), -1, order = 'F')

    
    return np.reshape(K*gradE, (-1))


def hessianOuterDissipationEnergy(pos, K, xA, xB=None, a_weight=1., b_weight=-0.95, mass_factor=None, timedict=None):
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
        add_block(0, 0, compute_hessian_external_edge_dissipation(
            pos_reshaped[1], pos_reshaped[0], a_weight, b_weight, 
            mass_factor=mass_factor, timedict=timedict))
    
    if xB is not None:
        add_block((num_freeShapes-1)*num_local_dof, (num_freeShapes-1)*num_local_dof, 
                 compute_hessian_external_edge_dissipation(
                     pos_reshaped[-2], pos_reshaped[-1], a_weight, b_weight, 
                     mass_factor=mass_factor, timedict=timedict))
    
    if xA is None:
        # Diagonal and mixed blocks combined in single loop
        for k in range(0, num_freeShapes-1):
            # Diagonal blocks
            add_block((k+1)*num_local_dof, (k+1)*num_local_dof, 
                     compute_hessian_external_edge_dissipation(
                         pos_reshaped[k+1], pos_reshaped[k], a_weight, b_weight, 
                         mass_factor=mass_factor, timedict=timedict))
            add_block(k*num_local_dof, k*num_local_dof, 
                     compute_hessian_external_edge_dissipation(
                         pos_reshaped[k], pos_reshaped[k+1], a_weight, b_weight, 
                         mass_factor=mass_factor, timedict=timedict))
            
            # Mixed blocks
            add_block(k*num_local_dof, (k+1)*num_local_dof, 
                     compute_hessian_mixed_external_edge_dissipation(
                         pos_reshaped[k], pos_reshaped[k+1], a_weight, b_weight, 
                         mass_factor=mass_factor, timedict=timedict))
            add_block((k+1)*num_local_dof, k*num_local_dof, 
                     compute_hessian_mixed_external_edge_dissipation(
                         pos_reshaped[k+1], pos_reshaped[k], a_weight, b_weight, 
                         mass_factor=mass_factor, timedict=timedict))
    
    if xA is not None:
        # Diagonal and mixed blocks combined in single loop
        for k in range(0, num_freeShapes-1):
            # Diagonal blocks
            add_block((k+1)*num_local_dof, (k+1)*num_local_dof, 
                     compute_hessian_external_edge_dissipation(
                         pos_reshaped[k+2], pos_reshaped[k+1], a_weight, b_weight, 
                         mass_factor=mass_factor, timedict=timedict))
            add_block(k*num_local_dof, k*num_local_dof, 
                     compute_hessian_external_edge_dissipation(
                         pos_reshaped[k+1], pos_reshaped[k+2], a_weight, b_weight, 
                         mass_factor=mass_factor, timedict=timedict))
            
            # Mixed blocks
            add_block(k*num_local_dof, (k+1)*num_local_dof, 
                     compute_hessian_mixed_external_edge_dissipation(
                         pos_reshaped[k+1], pos_reshaped[k+2], a_weight, b_weight, 
                         mass_factor=mass_factor, timedict=timedict))
            add_block((k+1)*num_local_dof, k*num_local_dof, 
                     compute_hessian_mixed_external_edge_dissipation(
                         pos_reshaped[k+2], pos_reshaped[k+1], a_weight, b_weight, 
                         mass_factor=mass_factor, timedict=timedict))
    rows = rows[:current_idx]
    cols = cols[:current_idx]
    data = data[:current_idx]
    # Build sparse matrix from COO format
    Hess = scipy.sparse.coo_matrix((data, (rows, cols)), 
                                    shape=(num_freeShapes * num_local_dof, num_freeShapes * num_local_dof))
    # Convert to CSR for efficient operations
    Hess = Hess.tocsr()
    return K * Hess