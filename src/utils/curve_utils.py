import numpy as np
import utils 
from numba import jit 

def compute_lines(positions):
    '''Computes an array of tuples [(0,1), (1,2),..., (len(positions)-1, len(positions))] of connectivity markers'''
    lines = np.zeros((positions.shape[0]-1,2))

    lines[:,0] = np.arange(0, positions.shape[0]-1, 1)
    lines[:,1] = np.arange(1,  positions.shape[0], 1)

    return np.array(lines)

####################################################
##### Utils for processing shape generation ########
####################################################
def rot_to_xy_plane(curve):
    '''
    Rotates planar curve to xy-plane such that curve normal is given by [0.,0.,1]
    (assumes curve has at least three points)
    '''
    normal = np.cross(curve[1] - curve[0], curve[-1] - curve[1])
    if np.linalg.norm(normal) == 0:
        """
        curve is a straight line
        """
        linedir  = curve[1] -curve[0]
        normal = np.array([-linedir[1] ,linedir[0], 0])

    normal = normal/np.linalg.norm(normal)

    conv_normal = np.array([0.,0.,1.])
    v = np.cross(normal, conv_normal)
    c = np.dot(normal, conv_normal)
    s = np.linalg.norm(v)
    if s == 0:
        "already aligned"
        return curve
    M = utils.assemble_Ab(v)[0]
    rotation_matrix = np.eye(3) + M + M.dot(M) * (1-c)

    return utils.dot_mat(rotation_matrix, curve)

def scale_curve_to_unit_length(curve):
    """
    Scale a curve such that it has total edge length 1.

    Parameters:
    curve (array-like): Input array of shape (N, 3) representing the vertices of the curve.

    Returns:
    array-like: Scaled curve with total edge length 1.
    """
    total_length = np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()

    # Scale each vertex of the curve
    scaled_curve = curve / total_length

    return scaled_curve

def compute_curve_length(curve_points):
    '''
    Computes total arc length along the curve

    Parameters:
    curve_points (array-like): Input curve points with shape (N, 3).
    '''
    cumulative_lengths = np.zeros(len(curve_points))
    for i in range(1, len(curve_points)):
        segment_length = np.linalg.norm(curve_points[i] - curve_points[i - 1])
        cumulative_lengths[i] = cumulative_lengths[i - 1] + segment_length

    return cumulative_lengths[-1]


### Resample for constant edge length routine
def distance(a, b):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(np.sum((b - a)**2))

def linear_curve_length(points):
    """Calculate the total length of the linear interpolation through a vector of points."""
    total_length = 0
    for i in range(len(points) - 1):
        total_length += distance(points[i], points[i + 1])
    return total_length

def uniform_linear_interpolation(source):
    """Perform uniform linear interpolation between the points in the source vector."""
    target_count = len(source)
    if target_count < 2:
        return []

    total_length = linear_curve_length(source)
    segment_length = total_length / (target_count - 1)
    result = [source[0]]  # Add the first point

    start = source[0]
    for i in range(1, target_count - 1):
        next_offset = segment_length * i
        src_segment_offset = 0
        src_segment_length = 0

        # Find the segment that contains the next offset
        for j in range(len(source) - 1):
            src_segment_length = distance(source[j], source[j + 1])
            if src_segment_offset + src_segment_length < next_offset:
                src_segment_offset += src_segment_length
                start = source[j + 1]
            else:
                break

        part_offset = next_offset - src_segment_offset
        part_ratio = part_offset / src_segment_length
        interpolated_point = (
            start[0] + part_ratio * (source[j + 1][0] - start[0]),
            start[1] + part_ratio * (source[j + 1][1] - start[1]),
            start[2] + part_ratio * (source[j + 1][2] - start[2])
        )
        result.append(interpolated_point)

    result.append(source[-1])  # Add the last point
    return np.array(result)


def resample_iterations(vertices, num_iter):
    for i in range(num_iter):
        #vertices = resample_for_equal_edge_length(vertices)
        vertices = uniform_linear_interpolation(vertices)

    return vertices

#########################################
##### Initialize shape Functions ########
#########################################    

def sin_with_phase_shift(phaseshift=0., freq=1., amplitude=1., num_samples=25):
    """
    Generate 3D vector positions of sine sampling for an interval of 0 to 2π.

    Parameters:
    phaseshift (float): phaseshift of sine
    freq (float): frequency of sine
    amplitude (float): amplitude of sine
    num_samples (int): Number of samples to generate.

    Returns:
    array-like: Array of 3D vector positions.
    """
    x = np.linspace(0, 2 * np.pi, num_samples)
    y = amplitude * np.sin( freq * x - phaseshift )
    z = np.zeros_like(x)  # Z-coordinate is set to zero in 2D scenario
    positions = np.column_stack((x, y, z))
    
    #return resample_for_equal_edge_length(scale_curve_to_unit_length(positions))
    return scale_curve_to_unit_length( resample_iterations(positions, 3) )


def serpenoid_shapes(t=0., sigma=0.5, center=np.array([0.0,0.0]), theta=0., a=1., wavelength=1., num_samples=25):
    ds = 1. * 1. / num_samples
    b = sigma * a
    arc_length_samples = np.linspace(0.0, 1.0, num_samples-1)

    # compute [ w1(t), w2(t) ] for a given time t
    w1_t = a * np.cos(2.0 * np.pi * t)
    w2_t = b * np.sin(2.0 * np.pi * t)
    w_t = np.array([ w1_t * np.cos(theta) - w2_t * np.sin(theta) , w1_t * np.sin(theta) + w2_t * np.cos(theta) ])

    # integrate kappa to phi ( integration constant set to 0)
    phi = np.zeros(num_samples-1)
    for s in range(1, num_samples-1):
        phi[s] = ( - w_t[0] * ( np.cos( 2 * np.pi * arc_length_samples[s] / wavelength ) ) + w_t[1] * np.sin( 2 * np.pi * arc_length_samples[s] ) ) / wavelength

    # integrate Tangent directions of gamma 
    gamma = np.zeros((num_samples-1, 3))
    for s in range(1, num_samples-1):
        gamma[s] = gamma[s-1] + ds * np.array( [ np.cos( phi[s] ) , 
                                                 np.sin( phi[s] ), 
                                                 0. ] )
        
    return gamma

### Normalizing curve positions
def pca(X):
    """
    Perform Principal Component Analysis (PCA) on a given dataset.

    Parameters:
    X (array-like): Input data matrix of shape (N, d), where N is the number of samples and d is the number of features.

    Returns:
    array-like: The principal components of the data.
    """
    # Compute the covariance matrix
    cov_matrix = np.cov(X, rowvar=False)

    # Compute the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvectors

def orient_and_center_point_clouds(point_clouds):
    """
    Orient and center multiple 3D point clouds.
    
    Parameters:
    point_clouds (array-like): Input array of shape (T, N, 3) representing T many (N, 3) point clouds.
    
    Returns:
    array-like: Oriented and centered point clouds with the same shape as the input.
    """
    num_point_clouds = len(point_clouds)
    oriented_centered_clouds = np.zeros_like(point_clouds)

    for i in range(num_point_clouds):
        # Center the point cloud at the origin
        centered_cloud = point_clouds[i] - np.mean(point_clouds[i], axis=0)

        # Perform PCA to orient the point cloud consistently
        eigenvectors = pca(centered_cloud)
        oriented_cloud = np.dot(centered_cloud, eigenvectors)

        oriented_centered_clouds[i] = oriented_cloud

    return oriented_centered_clouds

def integrate_turning_angles(alpha, edge_lengths=None):
    '''
    alpha is a (N,) array of turning angles
    edge_lengths is a (N+1,) array of N+1 edge lengths
    return a (N,3) array of integrated point positions from turning angles
    with starting edge direction(1,0,0)

    returns a curve reconsturcted from angels and edge lengths normalized such that first edge starts at origin and points in e1 direction
    '''
    if edge_lengths is None:
        edge_length = 1. / ( float(len(alpha)) +1 )
        edge_lengths = np.ones(len(alpha+1)) * edge_length
    alpha_int = np.cumsum(alpha)

    positions = np.zeros((len(alpha)+2, 3))
    positions[1] = edge_lengths[0] * np.array([1., 0., 0.])
    for i in range(len(alpha)):
        positions[i+2] = edge_lengths[i] * np.array([ np.cos(alpha_int[i]), np.sin(alpha_int[i]), 0.0 ]) + positions[i+1]

    return positions

def extract_turning_angles(positions):
    '''
    positions is a (N, 3) array of point positions
    return a (N,) array of turning angles
    '''
    angles = np.zeros(len(positions) - 2)
    for i in range(1, len(positions) - 1):
        prev = (positions[i] - positions[i - 1]) / np.linalg.norm(positions[i] - positions[i - 1])
        next_ = (positions[i + 1] - positions[i]) / np.linalg.norm(positions[i + 1] - positions[i])

        s = np.sign(np.cross(prev, next_))[2]
        angles[i - 1] = np.arccos( np.clip( np.dot(prev, next_ ) , -1., 1.) ) * s # added a clip method to ensure values in [-1,1] interval which is the domain of arccos 

    return angles

def extract_edge_lengths(positions):
    '''
    positions is a (N, 3) array of point positions
    return a (N-1,) array of the edge lengths
    '''
    edge_lengths = np.zeros(len(positions) - 1)
    for i in range(len(positions) - 1):
        edge_lengths[i] = np.linalg.norm(positions[i+1] - positions[i])

    return edge_lengths

def normalize_position(positions):
    """
    Returns a given curve in the normalized position such that 
    first edge starts at origin and points in e1 direction
    """
    angles = extract_turning_angles(positions)
    edge_lengths = extract_edge_lengths(positions)

    return integrate_turning_angles(angles, edge_lengths)

def normalize_sequence(positions_array):
    """
    Returns a sequence of curves in the normalized positions such that 
    for each curve the first edge starts at origin and points in e1 direction
    """
    return np.array([ normalize_position(positions) for positions in positions_array ])



######################################
##### Compute tangent vectors ########
######################################
def compute_vertex_tangent_vectors(positions):
    """
    Compute tangent vectors at the vertices of the planar polygonal curve.

    Parameters:
    phaseshift (float): phaseshift of sine
    freq (float): frequency of sine
    amplitude (float): amplitude of sine
    num_samples (int): Number of samples to generate.
    
    Returns:
    array-like: Array of tangent vectors at the vertices.
    """
    tangent_vectors = np.zeros(positions.shape)
    
    # Compute tangent vectors for the interior vertices
    for i in range(1, len(positions) - 1):
        # Normalize the edges
        edge1 = positions[i] - positions[i - 1]
        edge2 = positions[i + 1] - positions[i]
        edge1 /= np.linalg.norm(edge1)
        edge2 /= np.linalg.norm(edge2)
        # Average of normalized edges
        tangent_vectors[i] = (edge1 + edge2) / 2.
        if np.linalg.norm(tangent_vectors[i]) == 0:
            print("degenerate shape")
        else:
            tangent_vectors[i]/= np.linalg.norm(tangent_vectors[i])
    
    # Tangent vectors for the first and last vertices
    tangent_vectors[0] = ( positions[1] - positions[0] ) / np.linalg.norm(positions[1]-positions[0])
    tangent_vectors[-1] = ( positions[-1] - positions[-2] ) / np.linalg.norm(positions[-1]-positions[-2])
    
    # Normalize the tangent vectors
    tangent_vectors /= np.linalg.norm(tangent_vectors, axis=1)[:, np.newaxis]
    
    return tangent_vectors


def compute_vertex_weight(positions):
    n = len(positions)
    weights = np.zeros((n,1))

    for i in range(n):
        if i == 0:
            # For the first vertex, use only half of the length of the next edge
            weights[i] = np.linalg.norm(positions[i+1] - positions[i]) / 2.
        elif i == n - 1:
            # For the last vertex, use only half of the length of the previous edge
            weights[i] = np.linalg.norm(positions[i] - positions[i-1]) / 2.
        else:
            # For interior vertices, compute the average of half lengths of both adjacent edges
            edge1_length = np.linalg.norm(positions[i] - positions[i-1])
            edge2_length = np.linalg.norm(positions[i+1] - positions[i])
            weights[i] = (edge1_length + edge2_length) / 2.

    return weights

def compute_edge_tangent_vectors(positions):
    """
    Compute tangent vectors of on the edges of a polygonal curve associated to vertices of the curve with vertices [p_0,...,p_N]
    
    Returns:
    array-like: (N-1,3) array of tangent vectors on the edges.
    """
    edges = np.diff(positions, axis = 0)
    tangent_vectors = edges/np.linalg.norm(edges, axis = 1)[:,None]
    return tangent_vectors

def compute_edge_weight(positions):
    """
    Compute edge weigths [w_0,...,w_{N-1}], i.e. edge-length of the next edge of a polygonal 
    curve associated to vertices of the curve with vertices [p_0,...,p_N]

    Returns:
    array-like: Array of weigths of the edges.
    """
    edge_length = np.diff(positions, axis = 0)
    weights = np.linalg.norm(edge_length, axis = 1)[:,None]
    return weights


@jit(nopython=True, parallel=False)
def compute_edge_tangent_vectors_jit(positions):
    """Compute tangent vectors for edges - JIT compiled version"""
    N = positions.shape[0]
    tangent_vectors = np.zeros((N-1, 3))
    for i in range(N-1):
        edge = positions[i+1] - positions[i]
        edge_length = np.sqrt(edge[0]**2 + edge[1]**2 + edge[2]**2)
        tangent_vectors[i] = edge / edge_length
    return tangent_vectors


@jit(nopython=True, parallel=False)
def compute_edge_weight_jit(positions):
    """Compute edge weights - JIT compiled version"""
    N = positions.shape[0]
    edge_weights = np.zeros((N-1, 1))
    for i in range(N-1):
        edge = positions[i+1] - positions[i]
        edge_length = np.sqrt(edge[0]**2 + edge[1]**2 + edge[2]**2)
        edge_weights[i, 0] = edge_length
    return edge_weights


