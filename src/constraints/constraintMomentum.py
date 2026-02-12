import numpy as np
import scipy 
import time
import numba

from utils import *

def create_horizontal_constraint(K, xA, xB, a_weight, b_weight, mass_factor=None, timedict=None):
    """
    Returns the horizontal constraint function for scipy solver.
    """
    num_total_constr = 6*(K)
    if xB is None:
        num_total_constr = 6*(K-1)
        if xA is None:
            num_total_constr = 6*(K-2)
    scipy_horizontal_constraint =  scipy.optimize.NonlinearConstraint(
                                    lambda x : path_horizontal_constraint(x,K,xA,xB,a_weight=a_weight, 
                                                                b_weight=b_weight, mass_factor=mass_factor, timedict=timedict),
                                    jac = lambda x :path_dhorizontal_constraint(x,K,xA,xB, a_weight=a_weight, 
                                                                        b_weight=b_weight,mass_factor=mass_factor, timedict=timedict),
                                    hess = lambda x,v :path_ddhorizontal_constraintV(x,v,K,xA,xB,a_weight=a_weight, 
                                                                            b_weight=b_weight, mass_factor=mass_factor, timedict=timedict),
                                    lb = np.zeros(num_total_constr), ub = np.zeros(num_total_constr))

    return scipy_horizontal_constraint

def path_horizontal_constraint(x,K, xA=None, xB=None, a_weight = 1., b_weight=-.95, mass_factor=None, timedict=None):
    start_time = time.time()

    x = x.reshape((K-1, -1))
    if (xA is not None) and (xB is None):
        constr_val = np.zeros((K-1, 6))
    elif xA is None:
        constr_val = np.zeros((K-2, 6))
    else:
        constr_val = np.zeros((K,6))

    if xA is not None:
        for i in range(1,K-1):
            constr_val[i] = mu_edge(x[i-1].reshape((-1,3), order = 'F'), x[i]-x[i-1], a_weight =a_weight, 
                                    b_weight=b_weight, mass_factor=mass_factor)  
        constr_val[0] = mu_edge(xA.reshape((-1,3), order = 'F'), x[0]-xA, a_weight=a_weight, 
                                b_weight=b_weight, mass_factor=mass_factor)
    else:
        for i in range(0,K-2):
            constr_val[i] = mu_edge(x[i].reshape((-1,3), order = 'F'), x[i+1]-x[i], a_weight =a_weight, 
                                    b_weight=b_weight, mass_factor=mass_factor)   
    if xB is not None:
        constr_val[-1] = mu_edge(x[-1].reshape((-1,3), order = 'F'), xB-x[-1], a_weight =a_weight, 
                                 b_weight=b_weight, mass_factor=mass_factor)    
    
    if timedict is not None:
        timedict["mu"] +=time.time() - start_time
    
    return constr_val.reshape(-1)

def path_dhorizontal_constraint(x, K, xA=None, xB=None, a_weight=1., b_weight=-.95, mass_factor=None, timedict=None):
    start_time = time.time()
    x = x.reshape((K-1, -1))
    num_local_dof = x.shape[1]
    num_freeShapes = K-1
    num_constr = 6
    num_total_constr = num_constr*num_freeShapes
    
    if xB is not None:
        num_total_constr += num_constr
    if xA is None:
        num_total_constr -= num_constr

    # Pre-reshape all x slices
    x_reshaped = x.reshape((K-1, -1, 3), order='F')
    
    # Pre-allocate numpy arrays instead of Python lists
    estimated_nnz = num_freeShapes * num_local_dof * num_local_dof  # rough estimate
    rows = np.empty(estimated_nnz, dtype=np.int32)
    cols = np.empty(estimated_nnz, dtype=np.int32)
    data = np.empty(estimated_nnz, dtype=np.float64)
    current_idx = 0
    
    def add_block(row_start, col_start, block_matrix):
        """Helper to add a dense block to COO arrays - optimized version"""
        nonlocal current_idx
        
        block_rows, block_cols = block_matrix.shape
        
        # Check if we need more space (with auto-resize)
        nnz = block_rows * block_cols  # pessimistic: assume all non-zero
        if current_idx + nnz > len(rows):
            new_size = max(len(rows) * 2, current_idx + nnz)
            rows.resize(new_size, refcheck=False)
            cols.resize(new_size, refcheck=False)
            data.resize(new_size, refcheck=False)
        
        # Flatten and add all at once
        block_flat = block_matrix.ravel()
        nonzero_mask = block_flat != 0
        nnz_actual = np.count_nonzero(nonzero_mask)
        
        row_indices = np.repeat(np.arange(block_rows), block_cols) + row_start
        col_indices = np.tile(np.arange(block_cols), block_rows) + col_start
        
        # Add only non-zeros
        rows[current_idx:current_idx+nnz_actual] = row_indices[nonzero_mask]
        cols[current_idx:current_idx+nnz_actual] = col_indices[nonzero_mask]
        data[current_idx:current_idx+nnz_actual] = block_flat[nonzero_mask]
        current_idx += nnz_actual
    
    if xA is not None:
        xA_reshaped = xA.reshape((-1, 3), order='F')
        dmu_s = dmu_edge(xA_reshaped, x[0]-xA, a_weight=a_weight, b_weight=b_weight, 
                        mass_factor=mass_factor, timedict=timedict)
        add_block(0, 0, dmu_s)

    if xA is None:
        for i in range(0, num_freeShapes-1):
            dmu_t = -dmu_edge(x_reshaped[i+1], x[i]-x[i+1], a_weight=a_weight, b_weight=b_weight, 
                             mass_factor=mass_factor, timedict=timedict)
            dmu_s = dmu_edge(x_reshaped[i], x[i+1]-x[i], a_weight=a_weight, b_weight=b_weight, 
                            mass_factor=mass_factor, timedict=timedict)
            
            # derivative w.r.t. x[i]
            add_block(num_constr*i, i*num_local_dof, dmu_t)
            # derivative w.r.t. x[i+1]
            add_block(num_constr*i, (i+1)*num_local_dof, dmu_s)

    if xB is not None and xA is not None:
        xB_reshaped = xB.reshape((-1, 3), order='F')
        dmu_t = -dmu_edge(xB_reshaped, x[-1]-xB, a_weight=a_weight, b_weight=b_weight, 
                         mass_factor=mass_factor, timedict=timedict)
        add_block(num_total_constr - num_constr, (K-2)*num_local_dof, dmu_t)

    if xA is not None:
        for i in range(1, num_freeShapes):
            dmu_t = -dmu_edge(x_reshaped[i], x[i-1]-x[i], a_weight=a_weight, b_weight=b_weight, 
                             mass_factor=mass_factor, timedict=timedict)
            dmu_s = dmu_edge(x_reshaped[i-1], x[i]-x[i-1], a_weight=a_weight, b_weight=b_weight, 
                            mass_factor=mass_factor, timedict=timedict)
        
            # derivative w.r.t. x[i-1]
            add_block(num_constr*i, (i-1)*num_local_dof, dmu_t)
            # derivative w.r.t. x[i]
            add_block(num_constr*i, i*num_local_dof, dmu_s)
    # Trim arrays to actual size
    rows = rows[:current_idx]
    cols = cols[:current_idx]
    data = data[:current_idx]        
    # Build sparse matrix from COO format
    T = scipy.sparse.coo_matrix((data, (rows, cols)), 
                                 shape=(num_total_constr, num_freeShapes * num_local_dof))
    # Convert to CSR for efficient operations
    T = T.tocsr()
    
    if timedict is not None:
        timedict["dmu"] += time.time() - start_time   

    return T

def path_ddhorizontal_constraintV(x,v,K, xA=None, xB=None, a_weight = 1., b_weight=-.95, 
                                  mass_factor=None, timedict=None):
    start_time = time.time()
    x = x.reshape((K-1, -1))
    num_local_dof = x.shape[1]
    num_freeShapes = K-1
    num_constr = 6
    
    # Pre-reshape all x slices to avoid repeated reshaping in the loop
    x_reshaped = x.reshape((K-1, -1, 3), order='F')
    
    # Pre-allocate numpy arrays instead of Python lists
    estimated_nnz = num_freeShapes * num_local_dof * num_local_dof  # rough estimate
    rows = np.empty(estimated_nnz, dtype=np.int32)
    cols = np.empty(estimated_nnz, dtype=np.int32)
    data = np.empty(estimated_nnz, dtype=np.float64)
    current_idx = 0
    
    def add_block(row_start, col_start, block_matrix):
        """Helper to add a dense block to COO arrays - optimized version"""
        nonlocal current_idx
        
        block_rows, block_cols = block_matrix.shape
        
        # Check if we need more space (with auto-resize)
        nnz = block_rows * block_cols  # pessimistic: assume all non-zero
        if current_idx + nnz > len(rows):
            new_size = max(len(rows) * 2, current_idx + nnz)
            rows.resize(new_size, refcheck=False)
            cols.resize(new_size, refcheck=False)
            data.resize(new_size, refcheck=False)
        
        # Flatten and add all at once
        block_flat = block_matrix.ravel()
        nonzero_mask = block_flat != 0
        nnz_actual = np.count_nonzero(nonzero_mask)
        
        row_indices = np.repeat(np.arange(block_rows), block_cols) + row_start
        col_indices = np.tile(np.arange(block_cols), block_rows) + col_start
        
        # Add only non-zeros
        rows[current_idx:current_idx+nnz_actual] = row_indices[nonzero_mask]
        cols[current_idx:current_idx+nnz_actual] = col_indices[nonzero_mask]
        data[current_idx:current_idx+nnz_actual] = block_flat[nonzero_mask]
        current_idx += nnz_actual
    
    v_t = v[-num_constr:]
    
    if xB is None:
        ddmu_t = ddmu_edgev(v_t, x_reshaped[-2], x[-1] - x[-2], 
                                a_weight=a_weight, b_weight=b_weight, 
                                mass_factor=mass_factor, timedict=timedict)
    else:
        xB_reshaped = xB.reshape((-1, 3), order='F')
        ddmu_t = -ddmu_edgev(v_t, xB_reshaped, x[-1] - xB, 
                                 a_weight=a_weight, b_weight=b_weight, 
                                 mass_factor=mass_factor, timedict=timedict)

    # Add ddmu_t block (done in both branches)
    add_block((K-2)*num_local_dof, (K-2)*num_local_dof, ddmu_t)

    # Determine starting index for loop
    start_idx = 0 if xA is None else 1
    
    if xA is not None:
        v_s = v[:num_constr]
        xA_reshaped = xA.reshape((-1, 3), order='F')
        ddmu_s = ddmu_edgev(v_s, xA_reshaped, x[0] - xA, 
                                a_weight=a_weight, b_weight=b_weight, 
                                mass_factor=mass_factor, timedict=timedict)
        add_block(0, 0, ddmu_s)

    # Unified loop
    for i in range(start_idx if start_idx == 1 else 1, num_freeShapes):
        v_idx = (i-1)*num_constr if xA is None else i*num_constr
        v_s = v[v_idx:v_idx + num_constr]

        dmu_t, dmu_ts = ddmu_edgeandmixedv(v_s, x_reshaped[i], x[i-1] - x[i], 
                                                a_weight=a_weight, b_weight=b_weight, 
                                                mass_factor=mass_factor, timedict=timedict)
        dmu_s, dmu_st = ddmu_edgeandmixedv(v_s, x_reshaped[i-1], x[i] - x[i-1], 
                                                a_weight=a_weight, b_weight=b_weight, 
                                                mass_factor=mass_factor, timedict=timedict)
        
        # All four blocks
        add_block((i-1)*num_local_dof, (i-1)*num_local_dof, -dmu_t)
        add_block(i*num_local_dof, i*num_local_dof, dmu_s)
        add_block((i-1)*num_local_dof, i*num_local_dof, dmu_st)
        add_block(i*num_local_dof, (i-1)*num_local_dof, -dmu_ts)
    
    # Trim arrays to actual size
    rows = rows[:current_idx]
    cols = cols[:current_idx]
    data = data[:current_idx]
    
    # Build sparse matrix from COO format
    H = scipy.sparse.coo_matrix((data, (rows, cols)), 
                                 shape=(num_freeShapes * num_local_dof, num_freeShapes * num_local_dof))
    # Convert to CSR for efficient operations downstream
    H = H.tocsr()

    if timedict is not None:
        timedict["ddmu"] += time.time() - start_time      
    return H

def mu_edge(positions, displacements, tangent_vectors=None, edge_weights=None, a_weight=1., b_weight=-.95, mass_factor=None):
    
    if displacements.shape is not (len(positions), 3):
        displacements = displacements.reshape((-1,3), order = 'F')
    
    # check for tangent vectors and vertex weigths of gamma 
    if tangent_vectors is None: 
        tangent_vectors = compute_edge_tangent_vectors(positions) 
    if edge_weights is None:
        edge_weights =  compute_edge_weight(positions)
    
    N = len(positions)
    if isinstance(b_weight,float):
        b_weight = b_weight*np.ones((N-1,1))
    if mass_factor is None:
        mass_factor = np.ones((N-1,1))

    # implicit objects
    positions_next = positions + displacements
    tangent_vectors_next = compute_edge_tangent_vectors(positions_next)
    edge_weights_next = compute_edge_weight(positions_next)

    midpoint_positions =  0.5*(positions[:-1] + positions[1:])    
    midpoint_positions_next =  0.5*(positions_next[:-1] + positions_next[1:])
    midpoint_displacements = midpoint_positions_next - midpoint_positions

    Bv = a_weight * midpoint_displacements + b_weight * utils.dot_vec(midpoint_displacements, tangent_vectors) * tangent_vectors
    Bv_next = a_weight * midpoint_displacements + b_weight * utils.dot_vec(midpoint_displacements, tangent_vectors_next) * tangent_vectors_next

    mu_lin = - np.sum( 0.5*mass_factor*(edge_weights * Bv + edge_weights_next * Bv_next) , axis = 0)
    mu_ang = - 0.5 * np.sum(mass_factor*( edge_weights_next * np.cross(midpoint_positions, Bv_next, axis=-1) + 
                            edge_weights *  np.cross(midpoint_positions_next, Bv, axis=-1)) , axis=0)
    
    return np.concatenate((mu_lin, mu_ang), axis=0)

def dmu_edge(positions, displacements, tangent_vectors=None, edge_weights=None, a_weight=1., b_weight=-.95, mass_factor=None, timedict = None):
    """
    derivative wrt positions next (displacements)
    """
   
    if displacements.shape is not (len(positions), 3):
        displacements = displacements.reshape((-1,3), order = 'F')
    # check for tangent vectors and vertex weigths of gamma 
    if tangent_vectors is None: 
        tangent_vectors = compute_edge_tangent_vectors(positions) 
    if edge_weights is None:
        edge_weights =  compute_edge_weight(positions)
    
    N = len(positions)
    I = np.identity(3)
    mu_jac = np.zeros((6,N,3))
    dlin = np.zeros((N,3,3))
    dang = np.zeros((N,3,3))

    if isinstance(b_weight,float):
        b_weight = b_weight*np.ones((N-1,1))
    if mass_factor is None:
        mass_factor = np.ones((N-1,1))

    # implicit objects
    positions_next = positions + displacements
    tangent_vectors_next = compute_edge_tangent_vectors(positions_next)
    edge_weights_next = compute_edge_weight(positions_next)

    # edge midpoint averaged objects
    midpoint_positions =  0.5*(positions[:-1] + positions[1:])    
    midpoint_positions_next =  0.5*(positions_next[:-1] + positions_next[1:])
    midpoint_displacements = midpoint_positions_next - midpoint_positions

    #I- Tnext \oplus Tnext (N-1,3,3)
    T_nc_normal = I - np.einsum('ij,ik->ijk', tangent_vectors_next, tangent_vectors_next)

    #BtV
    BVc_cc = edge_weights*(a_weight * midpoint_displacements + b_weight * utils.dot_vec(midpoint_displacements, tangent_vectors) * tangent_vectors)
    #B(t+1)V
    BnVc_cc = (a_weight * midpoint_displacements + b_weight * utils.dot_vec(midpoint_displacements, tangent_vectors_next) * tangent_vectors_next)
    ddT = - np.einsum('kil, km -> kilm' , T_nc_normal, tangent_vectors_next) - np.einsum('ki,kml -> kilm', tangent_vectors_next,T_nc_normal) #first m second l  dPi
    T_nc_outer = np.einsum('kilm, ki -> klm', -ddT, midpoint_displacements) #same as old definition of T_nc_outer

    a_lin = np.array([-.5*tangent_vectors_next[k][:, np.newaxis] *( BnVc_cc[k]) - .5*b_weight[k]*T_nc_outer[k] for k in range(N-1)])

    T_nc_vec = tangent_vectors_next[:,:,np.newaxis]*tangent_vectors_next[:, np.newaxis, : ] 
    T_cc_vec = tangent_vectors[:,:,np.newaxis]*tangent_vectors[:, np.newaxis, : ] 
    
    #B(\ring(v))
    B_T_cc = edge_weights[:,np.newaxis]*( a_weight*I+ b_weight[:,np.newaxis,:]*T_cc_vec )  
    B_T_nc = edge_weights_next[:,np.newaxis]*( a_weight*I+  b_weight[:,np.newaxis,:]*T_nc_vec )
    
    #\gamma x B(\ring(v))
    Pc_ncB_T_cc = np.cross(midpoint_positions_next[:,np.newaxis], B_T_cc)
    Pc_ccB_T_nc = np.cross(midpoint_positions[:,np.newaxis], B_T_nc)
    
    #\ring(v) x Bv
    IBVc_cc = np.cross(I, BVc_cc[:, np.newaxis, :])
    
    #.5*(B(t,t+1)) #Bvring
    b_lin = .5*(.5*( B_T_cc + B_T_nc))

    a_ang = .5*(Pc_ncB_T_cc + IBVc_cc + Pc_ccB_T_nc)

    b_ang = np.cross(midpoint_positions[:,np.newaxis,:], 2*a_lin)

    #deriv wrt k
    dlin[:-1,:,:] += mass_factor[:,np.newaxis,:]*(a_lin + b_lin)
    #deriv wrt k+1
    dlin[1:,:,:] += mass_factor[:,np.newaxis,:]*(-a_lin  + b_lin)


    dang[:-1,:,:] += mass_factor[:,np.newaxis,:]*(a_ang + b_ang)
    dang[1:,:,:] += mass_factor[:,np.newaxis,:]*(- b_ang + a_ang)
    
    for i in range(3):
            mu_jac[:,:,i] = np.concatenate((-dlin[:,i,:],-0.5*dang[:,i,:]), axis = 1 ).T

    mu_jac = mu_jac.reshape(6, -1, order='F')

    return mu_jac

def ddmu_edgev(v, positions, displacements, tangent_vectors=None, edge_weights=None, a_weight=1., b_weight=-.95, mass_factor=None, timedict = None):
    """
        second derivative of mu_edge, twice differentiated with respect to positions next, minus and swapped arguments for differentiating wrt positions
    """
    start_time =time.time()
    N = len(positions)
    I = np.identity(3)
    if displacements.shape is not (len(positions), 3):
        displacements = displacements.reshape((-1,3), order = 'F')
    mu_hess = np.zeros((6,3*N,3*N))
    if isinstance(b_weight,float):
        b_weight = b_weight*np.ones((N-1,1))
    if mass_factor is None:
        mass_factor = np.ones((N-1,1))

    positions_next = positions + displacements

    tangent_vectors = compute_edge_tangent_vectors_jit(positions)
    tangent_vectors_next = compute_edge_tangent_vectors_jit(positions_next)

    edge_weights = compute_edge_weight_jit(positions)
    edge_weights_next = compute_edge_weight_jit(positions_next)

    midpoint_positions =  0.5*(positions[:-1] + positions[1:])    
    midpoint_positions_next =  0.5*(positions_next[:-1] + positions_next[1:])
    midpoint_displacements = midpoint_positions_next - midpoint_positions

    Btnv = (a_weight * midpoint_displacements + b_weight * dot_vec(midpoint_displacements, tangent_vectors_next) * tangent_vectors_next)

    T_nc_vec, T_cc_vec = compute_outer_products(tangent_vectors_next, tangent_vectors)

    B_T_cc = edge_weights[:,np.newaxis]*( a_weight*I+ b_weight[:,np.newaxis,:]*T_cc_vec )  # B(\ring(v))
    
    dT, ddT = compute_projection_derivatives(tangent_vectors_next)

    ddbarB = compute_ddbarB_tensors(ddT, dT, tangent_vectors_next, b_weight, edge_weights_next)
   
    ddbarBv = 0.5*np.einsum('kilmp, ki -> kmpl',ddbarB, midpoint_displacements)   
    T_nc_outer = -contract_with_displacement(ddT, midpoint_displacements)

    dbarB = -ddT

    dotB = np.array([-tangent_vectors_next[k,np.newaxis,:]*( ( a_weight*I+ b_weight[k]*T_nc_vec )[k, :, :, np.newaxis]) - 
                     b_weight[k]*dbarB[k].swapaxes(1,2) for k in range(N-1)])

    for k in range(N-1):
        #all term should have shape (3,3,3)
        
        ##########linear momentum 
        #\dot\ring w \tilde B v
        a_a = ((0.5/edge_weights_next[k])*dT[k][:, :,np.newaxis]*(Btnv[k])).T 
        #\ring w \dot \tilde B v
        a_b = (0.5*tangent_vectors_next[k, :,np.newaxis]*(b_weight[k]/edge_weights_next[k])*T_nc_outer[k,:,np.newaxis]).T
        # \ring w \tilde B \dot v
        a_c = -0.25*tangent_vectors_next[k,:, np.newaxis] * (a_weight*I+ b_weight[k]*T_nc_vec )[k,:, np.newaxis]
        # \dot \ring \bar B v
        b_a = ddbarBv[k] 
        #w \ring \tilde B \dot v
        b_b = -0.25* b_weight[k] * dbarB[k] 
        # \dot B \ring v
        c_a = 0.25* dotB[k]

        #dlin[:-1,:,:] first k then k
        lin_kk =  a_a + a_b + a_c  + b_a + b_b +  c_a
        #dlin[:-1,:,:] first k then k+1
        lin_kkp = -a_a  - a_b  + a_c -b_a +  b_b -c_a
        
        # # ,kk
        mu_hess[:3,3*k:3*(k+1),3*(k):3*(k+1) ] += mass_factor[k]*lin_kk
        # # #first k then k+1
        mu_hess[:3,3*k:3*(k+1),3*(k+1):3*(k+2)] += mass_factor[k]*lin_kkp

        #dlin[1:,:,:] first k+1 then k
        lin_kpk =  -a_a -a_b -a_c   -b_a   -b_b +c_a
        #dlin[1:,:,:] first k+1 then k+1
        lin_kpkp =  +a_a + a_b -a_c   + b_a - b_b -c_a

        ## first k+1, then k
        mu_hess[:3,3*(k+1):3*(k+2),3*(k):3*(k+1)] += mass_factor[k]*lin_kpk
        # k+1,k+1
        mu_hess[:3,3*(k+1):3*(k+2),3*(k+1):3*(k+2)] += mass_factor[k]*lin_kpkp

        ##########angular momentum 
        #first k then k+1
        #eta x dot ring(Bv)
        ang_kk = np.cross(midpoint_positions[k], 2*lin_kk , axis = 0) 
        ang_kkp = np.cross(midpoint_positions[k], 2*lin_kkp, axis = 0) 

        #ring eta x B(dotv)
        ringetaBdot = np.cross(I[:,:,np.newaxis],  0.25*B_T_cc[k], axis = 0)
        ang_kk += ringetaBdot
        ang_kkp += ringetaBdot
        #dot eta x B(ring v)
        dotetaBring = ringetaBdot.swapaxes(-1,-2)
        ang_kk += dotetaBring
        ang_kkp += dotetaBring
        #first k+1 then k
        #eta x dot ring(Bv)
        ang_kpk = np.cross(midpoint_positions[k], 2*lin_kpk , axis = 0) 
        ang_kpkp = np.cross(midpoint_positions[k], 2*lin_kpkp, axis = 0)    
        #dot eta x B(ring v)
        ang_kpk += dotetaBring
        ang_kpkp +=dotetaBring
        #ring eta x B(dotv)
        ang_kpk += ringetaBdot
        ang_kpkp += ringetaBdot
        
        # # first k then k
        mu_hess[3:,3*k:3*(k+1),3*(k):3*(k+1) ] += 0.5*mass_factor[k]*ang_kk
        # # #first k then k+1
        mu_hess[3:,3*k:3*(k+1),3*(k+1):3*(k+2)]+= 0.5*mass_factor[k]*ang_kkp
        ## first k+1, then k
        mu_hess[3:,3*(k+1):3*(k+2),3*(k):3*(k+1)] += 0.5*mass_factor[k]*ang_kpk
        # k+1,k+1
        mu_hess[3:,3*(k+1):3*(k+2),3*(k+1):3*(k+2)] +=  0.5*mass_factor[k]*ang_kpkp

    mu_hess = np.einsum('i,ijk->jk', v, -mu_hess)

    if timedict is not None:
        timedict["ddmu_test"] += time.time() - start_time 
    return utils.cc_mat_jit(mu_hess)

def ddmu_edgeandmixedv(v, positions, displacements, tangent_vectors=None, edge_weights=None, 
                           a_weight=1., b_weight=-.95, mass_factor=None, timedict=None):
    """
        second derivative of mu_edge, returns hessian twice differentiated with respect to positions next and
          mixed hessian once differentiated with respect to positions next, once with respect to positions 
    
    """
    start_time = time.time()
    N = len(positions)
    I = np.identity(3)
    
    if displacements.shape != (len(positions), 3):
        displacements = displacements.reshape((-1, 3), order='F')
    
    mu_hess = np.zeros((6, 3*N, 3*N))
    mu_hess_mixed = np.zeros((6, 3*N, 3*N))
    
    if isinstance(b_weight, float):
        b_weight = b_weight * np.ones((N-1, 1))
    if mass_factor is None:
        mass_factor = np.ones((N-1, 1))
    
    positions_next = positions + displacements
    
    tangent_vectors = compute_edge_tangent_vectors_jit(positions)
    tangent_vectors_next = compute_edge_tangent_vectors_jit(positions_next)

    edge_weights = compute_edge_weight_jit(positions)
    edge_weights_next = compute_edge_weight_jit(positions_next)
    
    midpoint_positions = 0.5 * (positions[:-1] + positions[1:])
    midpoint_positions_next = 0.5 * (positions_next[:-1] + positions_next[1:])
    midpoint_displacements = midpoint_positions_next - midpoint_positions
      
    dot_result = dot_vec(midpoint_displacements, tangent_vectors_next)
    
    Btnv = (a_weight * midpoint_displacements + b_weight * dot_result * tangent_vectors_next)
    
    T_nc_vec, T_cc_vec = compute_outer_products(tangent_vectors_next, tangent_vectors)
    
    B_T_cc = edge_weights[:, np.newaxis] * (a_weight * I + b_weight[:, np.newaxis, :] * T_cc_vec)
    B_T_nc = edge_weights_next[:, np.newaxis] * (a_weight * I + b_weight[:, np.newaxis, :] * T_nc_vec)
    
    # Compute projection derivatives using JIT
    dT, ddT = compute_projection_derivatives(tangent_vectors_next)
    dTc, ddTc = compute_projection_derivatives(tangent_vectors)

    # Compute higher order derivatives
    ddbarB = compute_ddbarB_tensors(ddT, dT, tangent_vectors_next, b_weight, edge_weights_next)

    dbarBn = -ddT
    dbarBc = -ddTc
    
    # Compute dotB tensors
    dotBn = compute_dotB_tensors(tangent_vectors_next, T_nc_vec, b_weight, dbarBn)
    dotBc = compute_dotB_tensors(tangent_vectors, T_cc_vec, b_weight, dbarBc)

    dotBnv = contract_with_displacement(dotBn, midpoint_displacements)
    dotBcv = contract_with_displacement(dotBc, midpoint_displacements)
    T_nc_outer = -contract_with_displacement(ddT, midpoint_displacements)
    ddbarBv = 0.5*np.einsum('kilmp, ki -> kmpl',ddbarB, midpoint_displacements)   

    for k in range(N-1):

        # # Linear momentum terms
    
        a_a = ((0.5 / edge_weights_next[k, 0]) * dT[k][:, :, np.newaxis] * Btnv[k]).T
        a_b = (0.5 * tangent_vectors_next[k, :, np.newaxis] * (b_weight[k, 0] / edge_weights_next[k, 0]) * T_nc_outer[k, :, np.newaxis]).T
        a_c = -0.25 * tangent_vectors_next[k, :, np.newaxis] * (a_weight * I + b_weight[k, 0] * T_nc_vec)[k, :, np.newaxis]
        b_a = ddbarBv[k]
        b_b = -0.25 * b_weight[k, 0] * dbarBn[k]
        c_a = 0.25 * dotBn[k]
        
        lin_kk = a_a + a_b + a_c + b_a + b_b + c_a
        lin_kkp = -a_a - a_b + a_c - b_a + b_b - c_a
        lin_kk_mixed = a_c + b_b - c_a
        lin_kkp_mixed = a_c + b_b + c_a
        
        mu_hess[:3, 3*k:3*(k+1), 3*k:3*(k+1)] += mass_factor[k, 0] * lin_kk
        mu_hess[:3, 3*k:3*(k+1), 3*(k+1):3*(k+2)] += mass_factor[k, 0] * lin_kkp
        mu_hess_mixed[:3, 3*k:3*(k+1), 3*k:3*(k+1)] += mass_factor[k, 0] * lin_kk_mixed
        mu_hess_mixed[:3, 3*k:3*(k+1), 3*(k+1):3*(k+2)] += mass_factor[k, 0] * lin_kkp_mixed
        
        lin_kpk = -a_a - a_b - a_c - b_a - b_b + c_a
        lin_kpkp = a_a + a_b - a_c + b_a - b_b - c_a
        lin_kpk_mixed = -a_c - b_b - c_a
        lin_kpkp_mixed = -a_c - b_b + c_a
        
        mu_hess[:3, 3*(k+1):3*(k+2), 3*k:3*(k+1)] += mass_factor[k, 0] * lin_kpk
        mu_hess[:3, 3*(k+1):3*(k+2), 3*(k+1):3*(k+2)] += mass_factor[k, 0] * lin_kpkp
        mu_hess_mixed[:3, 3*(k+1):3*(k+2), 3*k:3*(k+1)] += mass_factor[k, 0] * lin_kpk_mixed
        mu_hess_mixed[:3, 3*(k+1):3*(k+2), 3*(k+1):3*(k+2)] += mass_factor[k, 0] * lin_kpkp_mixed
        
        # # Angular momentum
        ang_kk, ang_kkp, ang_kpk, ang_kpkp = optimized_angular_terms(
            midpoint_positions[k],
            (lin_kk, lin_kkp, lin_kpk, lin_kpkp),
            I,
            B_T_cc[k]
        )        

        mu_hess[3:, 3*k:3*(k+1), 3*k:3*(k+1)] += 0.5 * mass_factor[k, 0] * ang_kk
        mu_hess[3:, 3*k:3*(k+1), 3*(k+1):3*(k+2)] += 0.5 * mass_factor[k, 0] * ang_kkp
        mu_hess[3:, 3*(k+1):3*(k+2), 3*k:3*(k+1)] += 0.5 * mass_factor[k, 0] * ang_kpk
        mu_hess[3:, 3*(k+1):3*(k+2), 3*(k+1):3*(k+2)] += 0.5 * mass_factor[k, 0] * ang_kpkp
        
        # Mixed angular terms
        bang_kk, bang_kkp, Tang_kk, Tang_kkp, Aang_kk, Aang_kkp = optimized_mixed_angular_terms(
            midpoint_positions[k],
            midpoint_positions_next[k],
            I,
            B_T_cc[k],
            B_T_nc[k],
            dotBc[k],
            dotBn[k],
            dotBcv[k],
            dotBnv[k]
        )
        dotetaringBv = 0.5 * np.cross(I[:, :, np.newaxis], dotBcv[k], axis=0).swapaxes(-1, -2)
        bang_kk = dotetaringBv
        bang_kkp = dotetaringBv
        
        etaringBdotv = 0.25 * np.cross(midpoint_positions_next[k], dotBc[k], axis=0).swapaxes(-1, -2)
        bang_kk += etaringBdotv
        bang_kkp += etaringBdotv

        mu_hess_mixed[3:, 3*k:3*(k+1), 3*k:3*(k+1)] += 0.5 * mass_factor[k, 0] * (bang_kk + Tang_kk + Aang_kk)
        mu_hess_mixed[3:, 3*k:3*(k+1), 3*(k+1):3*(k+2)] += 0.5 * mass_factor[k, 0] * (bang_kkp + Tang_kkp + Aang_kkp)
        mu_hess_mixed[3:, 3*(k+1):3*(k+2), 3*k:3*(k+1)] += 0.5 * mass_factor[k, 0] * (-bang_kk + Tang_kk + Aang_kk)
        mu_hess_mixed[3:, 3*(k+1):3*(k+2), 3*(k+1):3*(k+2)] += 0.5 * mass_factor[k, 0] * (-bang_kkp + Tang_kkp + Aang_kkp)
    
    mu_hess = np.einsum('i,ijk->jk', v, -mu_hess)
    mu_hess_mixed = np.einsum('i,ijk->jk', v, -mu_hess_mixed)
    
    if timedict is not None:
        timedict["ddmu_test"] += time.time() - start_time
    
    return cc_mat_jit(mu_hess),cc_mat_jit(mu_hess_mixed)

@numba.jit(nopython=True, parallel=False)
def compute_ddbarB_tensors(ddT, dT, tangent_vectors, b_weight, edge_weights):
    """Compute ddbarB tensors"""
    N = tangent_vectors.shape[0]
    ddbarB = np.zeros((N, 3, 3, 3, 3))
    
    for k in range(N):
        factor = b_weight[k, 0] / edge_weights[k, 0]
        for i in range(3):
            for l in range(3):
                for m in range(3):
                    for p in range(3):
                        # ddbarB = ddT ⊗ T + dT ⊗ dT + dT ⊗ dT + T ⊗ ddT
                        val = (ddT[k, i, l, p] * tangent_vectors[k, m] +
                               dT[k, i, l] * dT[k, m, p] +
                               dT[k, i, p] * dT[k, m, l] +
                               tangent_vectors[k, i] * ddT[k, m, l, p])
                        ddbarB[k, i, l, m, p] = factor * val
    
    return ddbarB

@numba.jit(nopython=True, parallel=False)
def compute_dotB_tensors(tangent_vectors, T_vec, b_weight, dbarB):
    """
    Compute dotB tensors
    Original: -tangent[k, newaxis, :] * (a*I + b*T)[k, :, :, newaxis] - b * dbarB[k].swapaxes(1,2)
    
    Structure:
    - First term: outer product creates shape where dotB[i,j,m] = -tangent[m] * (a*I + b*T)[i,j]
    - Second term: dbarB[k].swapaxes(1,2) means dbarB[i,m,j] becomes dbarB[i,j,m]
    - So: dotB[i,j,m] = -tangent[m] * (a*I + b*T)[i,j] - b * dbarB[i,m,j]
    """
    N = tangent_vectors.shape[0]
    I = np.eye(3)
    a_weight = 1.0  # hardcoded as in original
    dotB = np.zeros((N, 3, 3, 3))
    
    for k in range(N):
        for i in range(3):
            for j in range(3):
                for m in range(3):
                    val = -tangent_vectors[k, m] * (a_weight * I[i, j] + b_weight[k, 0] * T_vec[k, i, j])
                    val -= b_weight[k, 0] * dbarB[k, i, m, j]
                    dotB[k, i, j, m] = val
    
    return dotB

@numba.jit(nopython=True)
def optimized_angular_terms(midpoint_pos, lin_terms, I, B_T_cc):
    """
    JIT-compiled computation of angular momentum terms
    
    Args:
        midpoint_pos: (3,) midpoint position
        lin_terms: tuple of (lin_kk, lin_kkp, lin_kpk, lin_kpkp) each (3, 3, 3)
        I: (3, 3) identity matrix
        B_T_cc: (3, 3) B matrix
    
    Returns:
        ang_kk, ang_kkp, ang_kpk, ang_kpkp: (3, 3, 3) angular terms
    """
    lin_kk, lin_kkp, lin_kpk, lin_kpkp = lin_terms
    
    # Compute common terms
    ringetaBdot =  cross_I_B_axis0(I,0.25*B_T_cc) #

    dotetaBring = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            dotetaBring[i, j, :] = ringetaBdot[i, :, j]  # swapaxes(-1, -2)
    
    # ang_kk
    ang_kk = cross_product_3x3x3(midpoint_pos, 2 * lin_kk)
    ang_kk = ang_kk + ringetaBdot + dotetaBring
    
    # ang_kkp
    ang_kkp = cross_product_3x3x3(midpoint_pos, 2 * lin_kkp)
    ang_kkp = ang_kkp + ringetaBdot + dotetaBring
    
    # ang_kpk
    ang_kpk = cross_product_3x3x3(midpoint_pos, 2 * lin_kpk)
    ang_kpk = ang_kpk + dotetaBring + ringetaBdot
    
    # ang_kpkp
    ang_kpkp = cross_product_3x3x3(midpoint_pos, 2 * lin_kpkp)
    ang_kpkp = ang_kpkp + dotetaBring + ringetaBdot
    
    return ang_kk, ang_kkp, ang_kpk, ang_kpkp

@numba.jit(nopython=True)
def optimized_mixed_angular_terms(midpoint_pos, midpoint_pos_next, I, B_T_cc, B_T_nc,
                                   dotBc, dotBn, dotBcv, dotBnv):
    """
    JIT-compiled computation of mixed angular momentum terms
    
    Args:
        midpoint_pos: (3,) current midpoint position
        midpoint_pos_next: (3,) next midpoint position
        I: (3, 3) identity matrix
        B_T_cc: (3, 3) B matrix (current)
        B_T_nc: (3, 3) B matrix (next)
        dotBc: (3, 3, 3) dotB current
        dotBn: (3, 3, 3) dotB next
        dotBcv: (3, 3) dotB contracted with v (current)
        dotBnv: (3, 3) dotB contracted with v (next)
    
    Returns:
        Terms for mixed angular momentum
    """
    # dotetaBringv
    temp =  cross_I_B_axis0(I,-0.25*B_T_cc)
    dotetaBringv = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            dotetaBringv[i, j, :] = temp[i, :, j]  # swapaxes(-1, -2)
    
    Aang_kk = dotetaBringv
    Aang_kkp = dotetaBringv
    
    temp = cross_I_B_axis0(I, dotBcv)
    dotetaringBv = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            dotetaringBv[i, j, :] = 0.5 * temp[i, :, j]  # swapaxes(-1, -2) * 0.5


    bang_kk = dotetaringBv.copy()
    bang_kkp = dotetaringBv.copy()
    
    # etaringBdotv
    temp = cross_product_3x3x3(midpoint_pos_next, dotBc)
    etaringBdotv = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            etaringBdotv[i, j, :] = 0.25 * temp[i, :, j]  # swapaxes(-1, -2) * 0.25
    
    bang_kk =  bang_kk +etaringBdotv #.copy()
    bang_kkp = bang_kkp +  etaringBdotv#.copy()
    
    # ringetadotBv
    ringetadotBv = 0.5 * cross_I_B_axis0(I, dotBnv)
    Tang_kk = ringetadotBv.copy()
    Tang_kkp = -ringetadotBv.copy()
    
    # ringetaBdotv
    ringetaBdotv = cross_I_B_axis0(I, 0.25 * B_T_nc)
    Tang_kk = Tang_kk + ringetaBdotv
    Tang_kkp = Tang_kkp + ringetaBdotv
    
    # etadotBringv
    etadotBringv = 0.5 * cross_product_3x3x3(midpoint_pos, dotBn)
    Tang_kk = Tang_kk - etadotBringv
    Tang_kkp = Tang_kkp + etadotBringv
    
    return bang_kk, bang_kkp, Tang_kk, Tang_kkp, Aang_kk, Aang_kkp