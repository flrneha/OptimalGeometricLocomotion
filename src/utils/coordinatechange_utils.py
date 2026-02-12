import numpy as np
from numba import jit 

@jit(nopython=True)
def cc_mat_jit(A):
    '''
    reorders the enteries of a given Matrix A of shape (3n,3n) corresponding to the 
    coordinate convention ['x1' 'y1' 'z1' 'x2' 'y2' 'z2' 'x3' 'y3' 'z3' 'x4' 'y4' 'z4']
    to the coordinate convention ['x1' 'x2' 'x3' 'x4' 'y1' 'y2' 'y3' 'y4' 'z1' 'z2' 'z3' 'z4']
    '''
    n = A.shape[0]
    Id_ = np.identity(n)

    Id = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            row = j//3 + (j % 3) * n//3
            Id[i,j] = Id_[i,row]
    
    B = Id @ (A @ Id.T)

    return B