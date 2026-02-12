import os
import numpy as np
import time
from types import SimpleNamespace
import matplotlib.pyplot as plt
import scipy
from pathlib import Path


from constraints import *
from energies import *
import utils

"""
Minimal script for isoholonomic periodic problem. Start and end shape are free. 
Specify desired rbm and space and time resolution.
"""

#folder to store results
working_dir = Path(__file__).parent
result_path = working_dir / "/results/"
timestr = time.strftime("%Y%m%d-%H%M%S")
result_path = working_dir / 'results' / timestr
result_path.mkdir(parents=True, exist_ok=True)

# Time steps of path
K = 11
num_freeShapes = K - 1  # First and last shapes are fixed

# Specify parameters
param = {
    'bending_weight': 0.001,      # Controls resistance to bending
    'membrane_weight': 1.0,        # Controls resistance to stretching
    'energy_weight': 10.0,          # Overall energy weight
    'a_weight': 1.0,               # Isotropic dissipation
    'b_weight': -0.95,             # Anisotropic dissipation (negative = easier tangential motion)
    'K': K,
    'mass_factor': None,         #allows to set vertex-wise mass, if None, it is set to 1 for all vertices
    'theta':0,
    'b': [0.6,0., 0.]
}

parameters = SimpleNamespace(**param)

# Define path to data
# Adjust this path based on your repository structure
data_path = working_dir.parent.parent / "data" / "BVP"

# Load start shape
xA, lines = read_obj_curve(data_path / 'startbvp.obj')
xA = xA[::2]
# we use only every 4th vertex for illustration

lines = compute_lines(xA)

N = len(xA)
print(f"Start shape has {len(xA)} vertices")
print(f"Shape dimensions: {xA.shape}")

# Create end shape by applying rigid body motion
# RBM format: [rx, ry, rz, angle, tx, ty, tz]
rbm = [0., 0., 1., 0., *parameters.b]  # Translation in x, rotation around z-axis
xB = applyRBM(xA, rbm)

parameters.center =comp_center(xB)
# Specify the plane of the 2D curves (xy-plane)
normals = np.array([0., 0., 1.])

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(xA[:, 0], xA[:, 1], 'o-', linewidth=1, markersize=5, label='possible start shape', color='blue')
ax.plot(xB[:, 0], xB[:, 1], 'o-', linewidth=1, markersize=5, label='prescribed rbm', color='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('possible start and end shapes')
ax.set_aspect('equal', 'box')
ax.grid(True, alpha=0.3)
ax.legend()
plt.savefig(result_path / 'possible_start_end.png')

#reshape to correct format
xA = xA.reshape(-1, order = 'F') 
xA = np.reshape(xA,(1,-1))
xB = xB.reshape(-1, order = 'F')
xB = np.reshape(xB,(1,-1))


#initialize with constant path (jumping in the middle)
x0 = np.concatenate([int((K-1)/2)*[xA], int((K-1)/2)*[xB]])
x0 = x0.reshape(-1)


utils.save_parameters(parameters, result_path)

# Define parameters for energy functions
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

#unconstrained energy minimization between xA and xB
print("Starting unconstrained optimization to minimize the path energy...")
print("Method: Newton-CG")
print("="*60)

start_time = time.time()

optResult = scipy.optimize.minimize(
    path_energy.pathEnergy,
    x0,
    args,
    jac=path_energy.gradientPathEnergy,
    hess=path_energy.hessianPathEnergy,
    method='Newton-CG',
    options={'maxiter': 1000, 'disp': True}
)

unconstrained_time = time.time() - start_time

print(f"\nUnconstrained optimization completed in {unconstrained_time:.2f} seconds")
print(f"Success: {optResult.success}")
print(f"Final energy: {optResult.fun:.6f}")
print(f"Iterations: {optResult.nit}")

# Use result as starting point for constrained optimization
x0 = optResult.get('x')

#for comparison we first compute the bvp problem with fixed xA, and xB
print("Starting constrained optimization for boundary value problem...")

print("="*60)
constr_horizontal = create_horizontal_constraint(
    K, xA, xB,
    parameters.a_weight,
    parameters.b_weight,
    parameters.mass_factor
)
start_time = time.time()

optResult = scipy.optimize.minimize(
    path_energy.pathEnergy,
    x0,
    args,
    jac=path_energy.gradientPathEnergy,
    hess=path_energy.hessianPathEnergy,
    method='trust-constr',
    options={'maxiter': 3000, 'disp': True},
    constraints={constr_horizontal}
)

unconstrained_time = time.time() - start_time

print(f"\nBoundary value problem completed in {unconstrained_time:.2f} seconds")
print(f"Success: {optResult.success}")
print(f"Final energy: {optResult.fun:.6f}")
print(f"Iterations: {optResult.nit}")

bvp = optResult.get('x')
# Reshape and add fixed start/end shapes
bvp = np.reshape(bvp, (num_freeShapes, -1)) 
bvp = np.concatenate((xA, bvp, xB))

# Reshape to 3D coordinates
bvp = bvp.reshape((bvp.shape[0], -1, 3), order='F')

#visualize the path
animate_shape_sequence(bvp, save=True, fps  = 3, savepath=result_path / 'bvp_result')


x0 = np.concatenate([xA.flatten(),x0, xB.flatten()])

xA = None
xB = None
K = bvp.shape[0]+1
num_freeShapes = K - 1 
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

# Define horizontal constraint (6 constraints per time step)
constr_horizontal = create_horizontal_constraint(
    K, xA, xB,
    parameters.a_weight,
    parameters.b_weight,
    parameters.mass_factor
)
#constraints last shape to be a rigid body motion of first shape with parameters theta and b
constr_periodicity = create_periodic_constraint(K, N, parameters.theta, parameters.b)
#constraint center of mass of last shape to be at specified position 
#(rule out equivalent solutions given by rigid body motion of entire path)
constr_center = create_center_constraint(K, parameters.center)
#constraints total length of first shape to one (rule out degenerate solutions)
constr_edge_length = create_edge_length_constraint_first_shape(K, bounds = [-1e-6, 1e-6])

print("\nStarting constrained optimization for isoholonomic periodic problem...")
print("Method: trust-constr")
print("="*60)

start_time = time.time()

optResult = scipy.optimize.minimize(
    path_energy.pathEnergy,
    x0,
    args,
    jac=path_energy.gradientPathEnergy,
    hess=path_energy.hessianPathEnergy,
    method='trust-constr',
    options={'maxiter': 2000, 'disp': True},
    constraints={constr_horizontal, constr_periodicity, constr_center, constr_edge_length}
)

constrained_time = time.time() - start_time

print(f"\nConstrained optimization completed in {constrained_time:.2f} seconds")
print(f"Success: {optResult.success}")
print(f"Final energy: {optResult.fun:.6f}")
print(f"Iterations: {optResult.nit}")

# Get optimized path
opt = optResult.get('x')

# Reshape and add fixed start/end shapes
pos = np.reshape(opt, (num_freeShapes, -1)) 
#pos = np.concatenate((xA, pos, xB))

# Reshape to 3D coordinates
periodic_path = pos.reshape((pos.shape[0], -1, 3), order='F')

start = periodic_path[0]
end = periodic_path[-1]
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(start[:, 0], start[:, 1], 'o-', linewidth=1, markersize=5, label='start shape', color='blue')
ax.plot(end[:, 0], end[:, 1], 'o-', linewidth=1, markersize=5, label='end shape', color='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('optimized start and end shapes')
ax.set_aspect('equal', 'box')
ax.grid(True, alpha=0.3)
ax.legend()
plt.savefig(result_path / 'optimized_start_end.png')

en_args = ( lines, normals, parameters.bending_weight, parameters.membrane_weight, K, xA,xB)

print("Energy of isoholonomic path with free periodic boundary conditions:")
print("="*60)
print("Outer Disspiation Energy ", outerDissipationPathEnergy(opt,K, xA, xB))
print("Inner Disspiation Energy ", innerDissipationPathEnergy(opt, *en_args))
print("="*60)

print("Energy of boundary value problem solution:")
print("="*60)
bvpt = bvp.reshape((bvp.shape[0], -1), order='F')
bvpt = bvpt.reshape(-1) 
print("Outer Disspiation Energy ", outerDissipationPathEnergy(bvpt,K, xA, xB))
print("Inner Disspiation Energy ", innerDissipationPathEnergy(bvpt, *en_args))
print("="*60)

path  = pos.reshape((pos.shape[0],-1,3), order = 'F')
for i in range(len(path)):
    file = result_path  / f'path{i}.obj'
    utils.write_obj_curve(file, path[i], utils.compute_lines(path[0]))
steps = 1
fps  = 3
visu_path = np.array([path[i][:,[0,1]] for i in np.arange(0,len(path), 1)])
utils.visualize_shape_sequence(visu_path[::steps], savepath = result_path / 'periodic_result')
utils.animate_shape_sequence(visu_path[::steps], fps = fps, savepath = result_path / 'periodic_result' )