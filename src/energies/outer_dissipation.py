import numpy as np
from utils import *
import time
from constraints import *

def apply_B_tensor(a_weight, b_weight, Tangent, Edge):
    return a_weight*Edge + b_weight*np.dot(Edge,Tangent)*Tangent

def compute_external_edge_dissipation(displacements, positions, tangent_vectors=None, edge_weights=None, 
                                      a_weight = 1., b_weight = -.95, mass_factor=None):

    N = len(positions)

    if isinstance(b_weight,float):
        b_weight = b_weight*np.ones((N-1,1))

    if mass_factor is None:
        mass_factor = np.ones((N-1,1))

    if tangent_vectors is None:
        tangent_vectors = compute_edge_tangent_vectors_jit(positions)
    if edge_weights is None:
        edge_weights = compute_edge_weight_jit(positions)
    if mass_factor is None:
        mass_factor = np.ones_like(edge_weights)

    positions_next = positions + displacements
    tangent_vectors_next = compute_edge_tangent_vectors_jit(positions_next)
    edge_weights_next = compute_edge_weight_jit(positions_next)

    # edge midpoint averaged objects
    midpoint_positions =  0.5*(positions[:-1] + positions[1:])    
    midpoint_positions_next =  0.5*(positions_next[:-1] + positions_next[1:])
    midpoint_displacements = midpoint_positions_next - midpoint_positions

    Bt = (a_weight * midpoint_displacements + b_weight * dot_vec(midpoint_displacements, tangent_vectors) * tangent_vectors)
    Btn = (a_weight * midpoint_displacements + b_weight * dot_vec(midpoint_displacements, tangent_vectors_next) * tangent_vectors_next)
    B_avg = .5*(edge_weights* Bt + edge_weights_next * Btn)

    summands = np.array([mass_factor[k]*B_avg[k].dot(midpoint_displacements[k]) for k in range(N-1) ])

    return 0.5*np.sum(summands)

def compute_gradient_external_edge_dissipation(displacements, positions, tangent_vectors=None, a_weight = 1., b_weight = -.95,mass_factor=None):
    """
    Gradient with respect to positions
    """
    N = len(positions)
    I = np.identity(3)
    grad = np.zeros((N,3))
    if isinstance(b_weight,float):
        b_weight = b_weight*np.ones((N-1,1))
    if mass_factor is None:
        mass_factor = np.ones((N-1,1))
    # implicit objects
    positions_next = positions + displacements

    tangent_vectors = compute_edge_tangent_vectors_jit(positions)
    tangent_vectors_next = compute_edge_tangent_vectors_jit(positions_next)

    edge_weights = compute_edge_weight_jit(positions)
    edge_weights_next = compute_edge_weight_jit(positions_next)
    if mass_factor is None:
        mass_factor = np.ones((positions.shape[0],1))
    # edge midpoint averaged objects
    midpoint_positions =  0.5*(positions[:-1] + positions[1:])
    midpoint_positions_next =  0.5*(positions_next[:-1] + positions_next[1:])
    midpoint_displacements = midpoint_positions_next - midpoint_positions

    Bt = (a_weight * midpoint_displacements + b_weight * dot_vec(midpoint_displacements, tangent_vectors) * tangent_vectors)
    Btn = (a_weight * midpoint_displacements + b_weight * dot_vec(midpoint_displacements, tangent_vectors_next) * tangent_vectors_next)
    B_avg = .5*(edge_weights* Bt + edge_weights_next * Btn)

    dT, ddT = compute_projection_derivatives(tangent_vectors)
    T_nc_outer = -contract_with_displacement(ddT, midpoint_displacements)

    a = -0.25*np.sum((tangent_vectors[:, :, np.newaxis] * Bt[:, np.newaxis, :]) * midpoint_displacements[:][:, np.newaxis, :], axis=2)
    b = -0.25*np.array([b_weight[k]*T_nc_outer[k].dot(midpoint_displacements[k]) for k in range(N-1)])
    c = -0.5*B_avg

    grad[:-1,:] = mass_factor*( a + b + c) #k=0, ... N-1

    grad[1:,:] += mass_factor*(-a  -b + c) #k=1, ... N
    return grad

def compute_hessian_external_edge_dissipation(positions, positions_next, a_weight = 1., b_weight=-0.95, mass_factor=None,timedict =None):

    N = len(positions)
    I = np.identity(3)

    if isinstance(b_weight,float):
        b_weight = b_weight*np.ones((N-1,1))
    if mass_factor is None:
        mass_factor = np.ones((N-1,1))
    start_time = time.time()

    hess = np.zeros((3*N, 3*N))

    tangent_vectors = compute_edge_tangent_vectors_jit(positions)
    tangent_vectors_next = compute_edge_tangent_vectors_jit(positions_next)

    edge_weights = compute_edge_weight_jit(positions)
    edge_weights_next = compute_edge_weight_jit(positions_next)

    # edge midpoint averaged objects
    midpoint_positions =  0.5*(positions[:-1] + positions[1:])
    midpoint_positions_next =  0.5*(positions_next[:-1] + positions_next[1:])
    midpoint_displacements = midpoint_positions_next - midpoint_positions

    Btv = (a_weight * midpoint_displacements + b_weight * dot_vec(midpoint_displacements, tangent_vectors) * tangent_vectors)

    dT, ddT = compute_projection_derivatives(tangent_vectors)
    T_nc_outer = -contract_with_displacement(ddT, midpoint_displacements)

    star1_b = np.array([(b_weight[k]/edge_weights[k])*T_nc_outer[k].dot(midpoint_displacements[k]) for k in range(N-1)])
    T_nc_vec, T_cc_vec = compute_outer_products(tangent_vectors_next, tangent_vectors)

    #B(\ring(v))
    B_T_cc = edge_weights[:,np.newaxis]*( a_weight*I+ b_weight[:,np.newaxis,:]*T_cc_vec )
    B_T_nc = edge_weights_next[:,np.newaxis]*( a_weight*I+ b_weight[:,np.newaxis,:]*T_nc_vec )
    c_c = 0.25*( 0.5*(B_T_cc + B_T_nc))


    b_a = np.array([ 0.5*((b_weight[k]/edge_weights[k])*T_nc_outer[k]
                          * tangent_vectors[k].dot(midpoint_displacements[k])).T for k in range(N-1)])
    b_b = np.array([0.5*b_weight[k] * 0.5 *( dT[k] * tangent_vectors[k].dot(midpoint_displacements[k])) for k in range(N-1)])
    b_c = np.array([0.5*b_weight[k] * dT[k].dot(midpoint_displacements[k])[:, np.newaxis] 
                    * ((1./edge_weights[k])*dT[k].dot(midpoint_displacements[k])) for k in range(N-1)])
    b_d = np.array([0.5*b_weight[k] * dT[k].dot(midpoint_displacements[k])[:, np.newaxis] 
                    * (0.5*tangent_vectors[k]) for k in range(N-1)])

    a_c = 0.25 * tangent_vectors[:, :, np.newaxis] * Btv[:, np.newaxis]
    a_b = 0.25*tangent_vectors[:,:, np.newaxis]*star1_b[:,np.newaxis]

    c_b = 0.25* (b_weight[:,np.newaxis,:]) * T_nc_outer.swapaxes(-1,-2)

    b = -b_a + b_b + b_c + b_d
    b_kpk = b_a + b_b -b_c +b_d

    for k in range(N-1):

        a_a = (0.25/(edge_weights[k])*dT[k]*(Btv[k].dot(midpoint_displacements[k])))

        #star 1 kk
        hess[3*k:3*(k+1),3*(k):3*(k+1)] += mass_factor[k]*(a_a+a_b[k]+a_c[k] +b[k] + a_c[k].T+c_b[k]+c_c[k])
        #star 1 first k then k+1
        hess[3*k:3*(k+1),3*(k+1):3*(k+2)] += mass_factor[k]*(-a_a-a_b[k]+a_c[k] +b_kpk[k] - a_c[k].T-c_b[k]+c_c[k])

        #star 3 first k+1, then k
        hess[3*(k+1):3*(k+2),3*(k):3*(k+1)] +=  mass_factor[k]*(-a_a - a_b[k] - a_c[k]  -b[k]  +a_c[k].T + c_b[k] + c_c[k])
        #star 3 k+1,k+1
        hess[3*(k+1):3*(k+2),3*(k+1):3*(k+2)] +=  mass_factor[k]*(a_a+a_b[k]-a_c[k]   -b_kpk[k] - a_c[k].T-c_b[k]+c_c[k])

    if timedict is not None:
       timedict["hessloop"] += time.time() - start_time
    return cc_mat_jit(hess)

def compute_hessian_mixed_external_edge_dissipation(positions, positions_next, a_weight = 1., b_weight=-0.95, mass_factor=None,timedict =None):
    """
    first deriv w.r.t. positions
    second deriv w.r.t positions_next
    """
    start_time = time.time()

    N = len(positions)
    I = np.identity(3)

    if isinstance(b_weight,float):
        b_weight = b_weight*np.ones((N-1,1))
    if mass_factor is None:
        mass_factor = np.ones((N-1,1))

    hess = np.zeros((3*N, 3*N))

    tangent_vectors = compute_edge_tangent_vectors_jit(positions)
    tangent_vectors_next = compute_edge_tangent_vectors_jit(positions_next)

    edge_weights = compute_edge_weight_jit(positions)
    edge_weights_next = compute_edge_weight_jit(positions_next)

    # edge midpoint averaged objects
    midpoint_positions =  0.5*(positions[:-1] + positions[1:])
    midpoint_positions_next =  0.5*(positions_next[:-1] + positions_next[1:])
    midpoint_displacements = midpoint_positions_next - midpoint_positions

    Btv = (a_weight * midpoint_displacements + b_weight * dot_vec(midpoint_displacements, tangent_vectors) * tangent_vectors)
    Btnv = (a_weight * midpoint_displacements + b_weight * dot_vec(midpoint_displacements, tangent_vectors_next) * tangent_vectors_next)

    T_nc_normal = I - np.einsum('ij,ik->ijk', tangent_vectors, tangent_vectors)

    T_nc_normal_next, ddTn = compute_projection_derivatives(tangent_vectors_next)
    T_nc_outer_next_swapped = np.einsum('kilm, ki -> kml', -ddTn, midpoint_displacements) #same as old definition of T_nc_outer

    T_nc_vec, T_cc_vec = compute_outer_products(tangent_vectors_next, tangent_vectors)

    #B(\ring(v))
    B_T_cc = edge_weights[:,np.newaxis]*( a_weight*I+ b_weight[:,np.newaxis,:]*T_cc_vec )
    B_T_nc = edge_weights_next[:,np.newaxis]*( a_weight*I+ b_weight[:,np.newaxis,:]*T_nc_vec )
    c_c = 0.25*( 0.5*(B_T_cc + B_T_nc))

    a_c = 0.25 * tangent_vectors[:, :, np.newaxis] * Btv[:, np.newaxis]

    b_b  = np.array([ 0.5*b_weight[k] * 0.5 *( T_nc_normal[k] * tangent_vectors[k].dot(midpoint_displacements[k])) for k in range(N-1)])
    b_d = np.array([ 0.5*b_weight[k] * T_nc_normal[k].dot(midpoint_displacements[k])[:, np.newaxis] * (0.5*tangent_vectors[k]) for k in range(N-1)])

    c_b = 0.25* (b_weight[:,np.newaxis,:]) * T_nc_outer_next_swapped
    c_a = (0.25*tangent_vectors_next[:,:, np.newaxis] * Btnv[:,np.newaxis]).swapaxes(-1,-2)

    first_terms = a_c + b_d + b_b
    second_terms = c_a + c_b


    for k in range(N-1):

        #star 1 k
        hess[3*k:3*(k+1),3*(k):3*(k+1)] += mass_factor[k]*(-first_terms[k]  +second_terms[k] -c_c[k])
        #star 1 k+1
        hess[3*k:3*(k+1),3*(k+1):3*(k+2)] += mass_factor[k]*(-first_terms[k] - second_terms[k] -c_c[k])

        #star 3 k-1
        hess[3*(k+1):3*(k+2),3*(k):3*(k+1)] += mass_factor[k]*( first_terms[k]  +second_terms[k] - c_c[k])
        #star 3 k
        hess[3*(k+1):3*(k+2),3*(k+1):3*(k+2)] +=mass_factor[k]*( first_terms[k] -second_terms[k] -c_c[k])

    if timedict is not None:
       timedict["hessloop"] += time.time() - start_time
    return cc_mat_jit(hess)