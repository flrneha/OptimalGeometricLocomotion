import numpy as np
from pathlib import Path

def save_parameters(parameters, path, file='parameter.txt'):
    path = Path(path)
    with (path / file).open('w') as f:
        print(parameters, file=f)
        
def read_obj_curve(filepath):
    """
    read a standard .obj file
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

def write_obj_curve(filepath, vertices, lines):

    precision = 16
    n_vertices = vertices.shape[0]
    n_lines = lines.shape[0] if lines is not None else 0

    with open(filepath, 'w') as f:
        f.write('OFF\n')
        f.write(f'{n_vertices} {n_lines} 0\n')
        f.write(f'g \n')

        for i in range(n_vertices):
            f.write(f'v {" ".join([f"{coord:.{precision}f}" for coord in vertices[i]])}\n')
        f.write(f'g \n')
        f.write(f'l {" ".join([str(lin) for lin in np.arange(n_vertices,0, -1)])}\n')

    return
