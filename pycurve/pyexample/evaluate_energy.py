
import pycurve 
import numpy as np
import os
from curveIO import*

##Snake example
working_dir = os.path.abspath(os.getcwd())
read_path = working_dir + "/data/"
fileA = "snake_0.obj"
verticesA, linesA = read_obj_curve(read_path + fileA)

fileB = "snake_30.obj"
verticesB, linesB = read_obj_curve(read_path + fileB)

bending_weight  =0.001
normal = np.array([0,1,0])
print("__Snake Example___")
print("Curve_Energy", pycurve.curve_energy(verticesA, verticesB, linesA, normal,bending_weight))


### just for testing
print("__explicit toy example___")
v_undef = np.array([[0, 0, 0],
                    [ 0.8723510161150685, 0.5118696929498748, 0],
                    [1.9817743198576525, 0.833924420257607, 0],
                    [3.3013734912641786, 0.42312004657397906, 0],
                    [4, 0 ,0]
                    ])

v_def = np.array([[0,0,0], [1,0,0],[2,0,0],[3,0,0],[4,0,0]])
edges = np.array([[0,1],[1,2],[2,3],[3,4]])
normal = np.array([0,0,1])

# Energies
print("Membrane energy:", pycurve.membrane_energy(v_undef, v_def, edges))
print("Bending energy:", pycurve.bending_energy(v_undef, v_def, edges, normal))

print("Membrane undeformed gradient:", 
      np.reshape(pycurve.membrane_undeformed_gradient(v_undef, v_def, edges), (-1,3), order='F'))
print("Bending undeformed gradient:", 
      np.reshape(pycurve.bending_undeformed_gradient(v_undef, v_def, edges, normal), (-1,3), order='F'))
print("Curve deformed hessian:", 
     pycurve.curve_deformed_hessian(v_undef, v_def, edges, normal, bending_weight).shape)

bending_weight  = 0.001
print("Curve energy:", pycurve.curve_energy(v_undef, v_def, edges, normal, bending_weight))