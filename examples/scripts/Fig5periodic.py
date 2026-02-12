import os
import numpy as np
import scipy.optimize
import time
from types import SimpleNamespace
import utils
from constraints import * 
from energies import *

#folder to store results
working_dir = Path(__file__).parent
result_path = working_dir / "/results/geoIllustrate/"
timestr = time.strftime("%Y%m%d-%H%M%S")
result_path = working_dir / 'results' / timestr
os.mkdir(result_path)

K = 17
param = {'bending_weight':0.01,'membrane_weight':1. ,'energy_weight':1., 'a_weight' :1., 'b_weight' : -.95, 
         'K' : K, 'center': [0.,0.,0.], 'mass_factor' : None, 'theta':0., 'b': np.array([1.5,0, 0.])}

parameters = SimpleNamespace(**param)

data_dir =  working_dir.parent.parent / "data/Fig5/"

xA, lines = utils.read_obj_curve(data_dir / 'start.obj')

rbm = [0.,0.,1.,0,*parameters.b]
xB = xA + applyRBM(xA, rbm)

xA = xA.reshape(-1, order = 'F')
xB = xB.reshape(-1, order = 'F')
xA = np.reshape(xA,(1,-1))
xB = np.reshape(xB,(1,-1))
N = int(xA.shape[1]/3)

#convention for normal
normals = np.array([0.,0.,1.])

#initialize with start and end shape
x0 = np.concatenate([int((K-1)/2)*[xA], int((K-1)/2)*[xB]])
x0 = x0.reshape(-1)


lines = utils.compute_lines(xA.reshape((-1,3), order ='F' )) # connectivity of curves

#set xB None for center of mass constraint
desired_length = None

parameters.b  = np.array([1.3327063 , 0.08961233, 0.        ])
parameters.center = np.array([1.3327063, 0.2862203, 0.       ])
center = np.array(parameters.center)

#in each level K gets doubled
refinement_levels = 4
utils.save_parameters(parameters, result_path)


#weight factor for outer dissipation energy
parameters.energy_weight = 10.
for level in range(refinement_levels):

    args = ( lines,normals, parameters.bending_weight, parameters.membrane_weight, K, xA, xB, 
            parameters.energy_weight,parameters.a_weight, parameters.b_weight, parameters.mass_factor)

    #define horizontal consraint (6 constraints per time step)
    constr_horizontal  = create_horizontal_constraint(K,xA,xB, parameters.a_weight, parameters.b_weight, parameters.mass_factor)
    constr_edge_length = create_edge_length_constraint_first_shape(K)
    constr_periodicity = create_periodic_constraint(K, N, parameters.theta, parameters.b)
    constr_center = create_center_constraint(K, parameters.center)


    if level < 1:
        #solve boundary value problem without constraints
        print('K: ',K)
        optResult  = scipy.optimize.minimize(path_energy.pathEnergy, x0, args,jac = path_energy.gradientPathEnergy,hess = path_energy.hessianPathEnergy, 
                                            method = 'Newton-CG', options = { 'maxiter' : 1000, 'disp':True})    
        #for periodic problem start and end shape None (free)
        xA = None
        xB = None   
    if level >= 1:

        print('K: ',K)
        optResult  = scipy.optimize.minimize(path_energy.pathEnergy, x0, args,jac = path_energy.gradientPathEnergy,hess = path_energy.hessianPathEnergy, 
                                            method = 'trust-constr', options = { 'maxiter' : 3000, 'disp':True},
                                            constraints = {constr_horizontal, constr_center, constr_periodicity, constr_edge_length}) 

    opt = optResult.get('x')


    print(optResult, flush=True)
    args = ( lines,normals, parameters.bending_weight, parameters.membrane_weight, K, xA,xB)

    num_freeShapes = K-1
    if xB is None and (xA is not None):
        pos = np.reshape(opt, (num_freeShapes, -1))
        pos = np.concatenate((xA, pos))
    elif xA is None:
        pos = np.reshape(opt, (num_freeShapes,-1))
    else:
        pos = np.reshape(opt, (num_freeShapes, -1)) 
        pos = np.concatenate((xA, pos, xB))
    np.save(result_path / f'{level}geodesicpositions{K}_{N}', pos)

    path  = pos.reshape((pos.shape[0],-1,3), order = 'F')

    utils.save_parameters(parameters, result_path)

    steps = 1
    fps  = 3
    visu_path = np.array([path[i][:,[0,1]] for i in np.arange(0,len(path), 1)])
    utils.visualize_shape_sequence(visu_path[::steps], savepath = result_path / f'geodesic_result{level}')
    utils.animate_shape_sequence(visu_path[::steps], fps = fps, savepath = result_path / f'geodesic_result{level}' )
    
    for i in range(len(visu_path)):
        #write obj files
        file = result_path  / f'path{level}_{i}.obj'
        utils.write_obj_curve(file, path[i], utils.compute_lines(path[0]))

    if level < refinement_levels -1:
        opt = optResult.get('x')
        opt = np.reshape(opt, (num_freeShapes, -1))
        x0 = opt

        K = x0.shape[0] +1
        x0 = x0.reshape(-1)
        x0 = x0.reshape((K-1, -1))
        # double time resolution
        x0 = np.repeat(x0, 2, axis=0)
        K = x0.shape[0] +1
        x0 = x0.reshape(-1)
