import numpy as np

def read_obj_curve(filepath):
    """
    read a standard .obj file with line data
    assumes vertex data is sorted 

    Output
    -------------------------
    vertices,lines : (n,3), (n,2) array of vertices coordinates
                    and indices for lines
    """
    with open(filepath, 'r') as f:

        vertices = []
        lines = []

        for line in f:
            line = line.strip()
            if line == '' or line[0] == '#':
                continue

            line = line.split()
            if line[0] == 'v':
                vertices.append([float(x) for x in line[1:]])
            elif line[0] == 'l':
                lines.append([int(x.split('/')[0]) - 1 for x in line[1:]])
    vertices = np.asarray(vertices)
    lines = np.zeros((vertices.shape[0]-1,2))

    lines[:,0] = np.arange(0, vertices.shape[0]-1, 1)
    lines[:,1] = np.arange(1,  vertices.shape[0], 1)
    
    return vertices , lines