import os
import numpy as np
import scipy.optimize
import time
from types import SimpleNamespace
import utils
from constraints import * 
from energies import *
import matplotlib.pyplot as plt 

#folder to store results
working_dir = Path(__file__).parent
result_path = working_dir / "/results/geoIllustrate/"
timestr = time.strftime("%Y%m%d-%H%M%S")
result_path = working_dir / 'results' / timestr
os.mkdir(result_path)

K = 17
param = {'bending_weight':0.01,'membrane_weight':1. ,'energy_weight':1., 'a_weight' :1., 'b_weight' : -.95, 
         'K' : K, 'center': [0.,0.,0.],'mass_factor' : None,
         'theta':0., 'b': np.array([1.,0, 0.])}

parameters = SimpleNamespace(**param)

data_dir =  working_dir.parent.parent / "data/Fig5/"

xA, lines = utils.read_obj_curve(data_dir / 'start.obj')
xB, lines = utils.read_obj_curve(data_dir / 'scurveend.obj')


fig, ax = plt.subplots()
ax.plot(xA[:, 0], xA[:, 1], label='Start Shape')
ax.plot(xB[:, 0], xB[:, 1], label='End Shape')
ax.set_aspect('equal', 'box')
ax.legend()
plt.savefig(result_path / 'possible_start_end_shapes.png')


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

#weight factor for outer dissipation energy
parameters.energy_weight = 10.

parameters.center = comp_center(xB)
displ = comp_center(xB) - comp_center(xA)
parameters.b = displ

center = np.array(parameters.center)

#in each level K gets doubled
refinement_levels = 3
save_parameters(parameters, result_path)

for level in range(refinement_levels):

    args = ( lines,normals, parameters.bending_weight, parameters.membrane_weight, K, xA, xB, 
            parameters.energy_weight,parameters.a_weight, parameters.b_weight, parameters.mass_factor)

    if level < 1:
        #solve boundary value problem without constraints
        print('K: ',K)
        optResult  = scipy.optimize.minimize(path_energy.pathEnergy, x0, args,jac = path_energy.gradientPathEnergy,hess = path_energy.hessianPathEnergy, 
                                            method = 'Newton-CG', options = { 'maxiter' : 1000, 'disp':True})     

        #for chi holonomic problem end shape None (free)
        xB = None
    if level >= 1:
        #define horizontal consraint (6 constraints per time step)
        constr_horizontal  = create_horizontal_constraint(K,xA,xB, parameters.a_weight, parameters.b_weight, parameters.mass_factor)
        # we want a pure discplacement therefor we constrain the center of the (free) last shape
        constr_center = create_center_constraint(K, parameters.center)

        print('K: ',K)
        optResult  = scipy.optimize.minimize(path_energy.pathEnergy, x0, args, jac = path_energy.gradientPathEnergy, hess = path_energy.hessianPathEnergy, 
                                            method = 'trust-constr', options = { 'maxiter' : 3000, 'disp':True},
                                            constraints = {constr_horizontal, constr_center}) 

    opt = optResult.get('x')

    print(optResult, flush=True)
    
    args = (lines, normals, parameters.bending_weight, parameters.membrane_weight, K, xA, xB)

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

    save_parameters(parameters, result_path)

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
        #double time resolution
        opt = optResult.get('x')
        opt = np.reshape(opt, (num_freeShapes, -1))
        x0 = opt

        K = x0.shape[0] +1
        x0 = x0.reshape(-1)
        x0 = x0.reshape((K-1, -1))
        x0 = np.repeat(x0, 2, axis=0)
        
        K = x0.shape[0] +1
        x0 = x0.reshape(-1)

