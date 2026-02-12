import os
import numpy as np
import time
from types import SimpleNamespace
import scipy
import matplotlib.pyplot as plt
from constraints import *
from energies import path_energy
import utils

"""
Minimal script for chi-holonomic problem (mixed boundary conditions) Input: a start shape and a desired displacement
"""
def PurcellShapes(theta_1, theta_2):
    
    # z-coordinate set to be zero
    P = np.zeros((4,3))
    P[0] = np.array([ - 0.6, 0.1, 0.]) - np.array([ np.cos(theta_2), np.sin(theta_2), 0. ])
    P[1] = np.array([ - 0.6, 0.1, 0.])
    P[2] = np.array([ 0.4, 0.1, 0.])
    P[3] = np.array([ 0.4, 0.1, 0.]) + np.array([ np.cos(theta_1), np.sin(theta_1), 0. ])
    P = utils.scale_curve_to_unit_length(P)
    return P

#folder to store results
working_dir = Path(__file__).parent
result_path = working_dir / "results/"
timestr = time.strftime("%Y%m%d-%H%M%S")
result_path = working_dir / 'results' / timestr
result_path.mkdir(parents=True, exist_ok=True)

#time steps of path
K = 15
num_freeShapes = K-1

#specify parameters
param = {
    'bending_weight': 0.01,      # Controls resistance to bending
    'membrane_weight': 1.0,        # Controls resistance to stretching
    'energy_weight': 10.0,          # Overall energy weight
    'a_weight': 1.0,               # Isotropic dissipation
    'b_weight': -0.95,             # Anisotropic dissipation (negative = easier tangential motion)
    'K': K,
    'mass_factor': None,         #allows to set vertex-wise mass, if None, it is set to 1 for all vertices
    'theta':0.3,                   # specify rotation degree for prescribed rbm
    'b': [0.2,0.1, 0.]            # specify translation for prescribed rbm
}

parameters = SimpleNamespace(**param)
# Adjust this path based on your repository structure
data_path = working_dir.parent.parent / "data"  /"BVP"

# Load start shape
xA = PurcellShapes(0.2, 0.4)
lines = compute_lines(xA)

N = len(xA)

# Create end shape by applying rigid body motion
# RBM format: [rx, ry, rz, angle, tx, ty, tz]
rbm = [0., 0., 1., parameters.theta, *parameters.b]  # Translation in x, rotation around z-axis
parameters.rbm = rbm
xB = applyRBM(xA, rbm)

parameters.center = comp_center(xB)
# Specify the plane of the 2D curves (xy-plane)
normals = np.array([0., 0., 1.])


fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(xA[:, 0], xA[:, 1], 'o-', linewidth=1, markersize=5, label='Start Shape', color='blue')
ax.plot(xB[:, 0], xB[:, 1], 'o-', linewidth=1, markersize=5, label='prescribed rbm', color='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('possible start and end shapes')
ax.set_aspect('equal', 'box')
ax.grid(True, alpha=0.3)
ax.legend()
plt.savefig(result_path / 'possible_start_end_shapes.png')


#reshape to correct format
xA = xA.reshape(-1, order='F') 
xB = xB.reshape(-1, order='F')
xA = np.reshape(xA, (1, -1))
xB = np.reshape(xB, (1, -1))
# Initialize with constant path (jumping in the middle)
x0 = np.concatenate([int((K-1)/2) * [xA], int((K-1)/2) * [xB]])
x0 = x0.reshape(-1)

#target shape is free
xB = None
num_freeShapes = K - 1 

utils.save_parameters(parameters, result_path)

#define arguments
args = (
    lines, normals,
    parameters.bending_weight,
    parameters.membrane_weight,
    K, xA, xB,
    parameters.energy_weight,
    parameters.a_weight,
    parameters.b_weight,
    parameters.mass_factor
)

#define horizontal consraint (6 constraints per time step)
constr_horizontal  = create_horizontal_constraint(K, xA, xB, parameters.a_weight, parameters.b_weight, 
                                       parameters.mass_factor)

constr_rbm = create_chi_constraint(K,parameters.rbm)
print("\nStarting constrained optimization...")
print("Method: trust-constr")
print("="*60)

start_time = time.time()

#optimization problem, trust constraint algorithm
optResult  = scipy.optimize.minimize(path_energy.pathEnergy, x0, args, 
                                     jac = path_energy.gradientPathEnergy, 
                                     hess = path_energy.hessianPathEnergy, 
                                     method = 'trust-constr', 
                                     options = { 'maxiter' : 2000, 'disp':True}, 
                                     constraints = {constr_horizontal, constr_rbm})  

constrained_time = time.time() - start_time

print(f"\nConstrained optimization completed in {constrained_time:.2f} seconds")
print(f"Success: {optResult.success}")
print(f"Final energy: {optResult.fun:.6f}")
print(f"Iterations: {optResult.nit}")

# Get optimized path
opt = optResult.get('x')
pos = np.reshape(opt, (num_freeShapes, -1)) 
pos = np.concatenate((xA, pos))


print(optResult, flush=True)
path  = pos.reshape((pos.shape[0],-1,3), order = 'F')

for i in range(len(path)):
    file = result_path  / f'path{i}.obj'
    utils.write_obj_curve(file, path[i], utils.compute_lines(path[0]))

np.save(result_path / 'geodesic_path.npy', path)

steps = 1
fps  = 3
visu_path = np.array([path[i][:,[0,1]] for i in np.arange(0,len(path), 1)])
utils.visualize_shape_sequence(visu_path[::steps], savepath = result_path / 'geodesic_result')
utils.animate_shape_sequence(visu_path[::steps], fps = fps, savepath = result_path / 'geodesic_result' )