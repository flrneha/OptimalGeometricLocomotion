import os
import numpy as np
import time
from types import SimpleNamespace
import matplotlib.pyplot as plt
import scipy
from pathlib import Path
from constraints import *
from energies import path_energy
from utils import * 

"""
Minimal script for BV problem. First and last shape are fixed.
"""


#folder to store results
working_dir = Path(__file__).parent
result_path = working_dir / "/results/"
timestr = time.strftime("%Y%m%d-%H%M%S")
result_path = working_dir / 'results' / timestr
result_path.mkdir(parents=True, exist_ok=True)

print("Results are saved in: ", result_path)
#time steps of path
K = 15
num_freeShapes = K-1

#specify parameters, anisotropy tensor for outer dissipation a_weight * I + b_weight* T \outer T
param = {'bending_weight':0.001,'membrane_weight':1. ,'energy_weight':10., 'a_weight' :1., 'b_weight' : -.95, 
         'K' : K, 'mass_factor':None}

parameters = SimpleNamespace(**param)

#define start and end shape, load from data
data_path = working_dir.parent.parent / "data" / "BVP"

# Load start shape
xA, lines = read_obj_curve(data_path / 'startbvp.obj')
xB, _ = read_obj_curve(data_path / 'endbvp.obj')

# Specify the plane of the 2D curves (xy-plane)
normals = np.array([0., 0., 1.])
#specify displacement between start and end shape
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(xA[:, 0], xA[:, 1], 'o-', linewidth=1, markersize=5, label='Start Shape', color='blue')
ax.plot(xB[:, 0], xB[:, 1], 'o-', linewidth=1, markersize=5, label='End Shape', color='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('start and end shapes')
ax.set_aspect('equal', 'box')
ax.grid(True, alpha=0.3)
ax.legend()
plt.savefig(result_path / 'start_end_shapes.png')

#reshape to correct format
xA = xA.reshape(-1, order = 'F') 
xB = xB.reshape(-1, order = 'F')
xA = np.reshape(xA,(1,-1))
xB = np.reshape(xB,(1,-1))

#number of vertices
N = xA.shape[0]

#initialize with constant path (jumping in the middle)
x0 = np.concatenate([int((K-1)/2)*[xA], int((K-1)/2)*[xB]])
x0 = x0.reshape(-1)


utils.save_parameters(parameters, result_path)

#define parameters
args = ( lines,normals, parameters.bending_weight, parameters.membrane_weight, K, xA, xB, 
        parameters.energy_weight,parameters.a_weight, parameters.b_weight, parameters.mass_factor)

#define horizontal consraint (6 constraints per time step)
constr_horizontal  = create_horizontal_constraint(K,xA,xB, parameters.a_weight, parameters.b_weight, 
                                                  parameters.mass_factor)

print("Starting unconstrained optimization...")
print("Method: Newton-CG")
print("="*60)
start_time = time.time()

#first compute minimizer of path energy without constraint
optResult  = scipy.optimize.minimize(path_energy.pathEnergy, x0, args,jac = path_energy.gradientPathEnergy,
                                        hess = path_energy.hessianPathEnergy, 
                                        method = 'Newton-CG', options = { 'maxiter' : 1000, 'disp':True})
unconstrained_time = time.time() - start_time

print(f"\nUnconstrained optimization completed in {unconstrained_time:.2f} seconds")
print(f"Success: {optResult.success}")
print(f"Final energy: {optResult.fun:.6f}")
print(f"Iterations: {optResult.nit}")

x0 = optResult.get('x')
print("\nStarting constrained optimization...")
print("Method: trust-constr")
print("="*60)

start_time = time.time()
#optimization problem, trust constraint algorithm
optResult  = scipy.optimize.minimize(path_energy.pathEnergy, x0, args, jac = path_energy.gradientPathEnergy,
                                        hess = path_energy.hessianPathEnergy, 
                                        method = 'trust-constr', options = { 'maxiter' : 3000, 'disp':True}, 
                                        constraints = {constr_horizontal})  

constrained_time = time.time() - start_time

print(f"\nConstrained optimization completed in {constrained_time:.2f} seconds")
print(f"Success: {optResult.success}")
print(f"Final energy: {optResult.fun:.6f}")
print(f"Iterations: {optResult.nit}")
opt = optResult.get('x')

pos = np.reshape(opt, (num_freeShapes, -1)) 
pos = np.concatenate((xA, pos, xB))

print(optResult, flush=True)
path  = pos.reshape((pos.shape[0],-1,3), order = 'F')

for i in range(len(path)):
    file = result_path  / f'path{i}.obj'
    utils.write_obj_curve(file, path[i], utils.compute_lines(path[0]))

steps = 1
fps  = 3 #framerate
visu_path = np.array([path[i][:,[0,1]] for i in np.arange(0,len(path), 1)])
utils.visualize_shape_sequence(visu_path[::steps], savepath = result_path / 'geodesic_result')
utils.animate_shape_sequence(visu_path[::steps], fps = fps, savepath = result_path / 'geodesic_result' )