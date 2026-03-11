import numpy as np

def sinc(theta):
    return np.sinc(theta / np.pi) if abs(theta) > 1e-8 else 1.0

def msinc(theta):
    if abs(theta) < 1e-8:
        return 1/6.0
    return (sinc(theta) - np.cos(theta)) / (theta ** 2)

def mcosc(theta):
    if abs(theta) < 1e-8:
        return 0.5
    return (1 - np.cos(theta)) / (theta ** 2)


    
    
def exponential_map(v, delta_t=0.04):
    v = np.array(v).flatten()
    v_dt = v * delta_t
    t = v_dt[:3]
    u = v_dt[3:]

    theta = np.linalg.norm(u)
    if theta < 1e-8:
        R = np.eye(3)
        dt = t
    else:
        ux, uy, uz = u
        skew_u = np.array([
            [0, -uz, uy],
            [uz, 0, -ux],
            [-uy, ux, 0]
        ])

        u_outer = np.outer(u, u)
        I = np.eye(3)

        sinc_theta = sinc(theta)
        msinc_theta = msinc(theta)
        mcosc_theta = mcosc(theta)

        R = I + (sinc_theta * skew_u) + (mcosc_theta * np.dot(skew_u, skew_u))

        # Translation term
        dt = np.array([
            t[0] * (sinc_theta + ux**2 * msinc_theta) +
            t[1] * (ux*uy*msinc_theta - uz*mcosc_theta) +
            t[2] * (ux*uz*msinc_theta + uy*mcosc_theta),

            t[0] * (ux*uy*msinc_theta + uz*mcosc_theta) +
            t[1] * (sinc_theta + uy**2 * msinc_theta) +
            t[2] * (uy*uz*msinc_theta - ux*mcosc_theta),

            t[0] * (ux*uz*msinc_theta - uy*mcosc_theta) +
            t[1] * (uy*uz*msinc_theta + ux*mcosc_theta) +
            t[2] * (sinc_theta + uz**2 * msinc_theta)
        ])


    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = dt
    return T

def log_map(T):
    R = T[:3, :3]
    t = T[:3, 3]

    # Compute angle theta from trace
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        w = np.zeros(3)
        V_inv = np.eye(3)
    else:
        lnR = (theta / (2 * np.sin(theta))) * (R - R.T)
        w = np.array([lnR[2, 1], lnR[0, 2], lnR[1, 0]])

        # Compute the inverse of the left Jacobian V_inv
        wx, wy, wz = w
        skew_w = np.array([
            [0, -wz, wy],
            [wz, 0, -wx],
            [-wy, wx, 0]
        ])
        I = np.eye(3)

        A = sinc(theta)
        B = mcosc(theta)
        C = msinc(theta)

        V_inv = I - 0.5 * skew_w + (1 / theta**2) * (1 - A / (2 * B)) * (skew_w @ skew_w)

    v = V_inv @ t
    return np.concatenate([v, w])



def random_rotation_matrix(std_dev_deg=5.0):
    """
    Generate a random rotation matrix using small-angle perturbations.

    Parameters:
        std_dev_deg (float): Standard deviation in degrees for the rotation.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    std_dev_rad = np.radians(std_dev_deg)
    rx, ry, rz = np.random.normal(0.0, std_dev_rad, size=3)

    # Rotation matrices around X, Y, Z axes
    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    rot_y = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    rot_z = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    # Combined rotation: R = Rz * Ry * Rx
    return rot_z @ rot_y @ rot_x



import numpy as np

def perturb_rotation_matrix(R, std_dev_deg=(5.0, 5.0, 5.0), rng=None):
    """
    Apply a small random rotation to an existing rotation matrix.

    Parameters:
        R (np.ndarray): Original 3x3 rotation matrix.
        std_dev_deg (float or tuple/list of 3 floats): Standard deviation(s) in degrees for rotations around X, Y, Z.
        rng (np.random.Generator): Optional random number generator for reproducibility.

    Returns:
        np.ndarray: A new 3x3 rotation matrix with a slight perturbation.
    """
    assert R.shape == (3, 3), "Input rotation matrix must be 3x3"

    if isinstance(std_dev_deg, (int, float)):
        std_dev_deg = (std_dev_deg, std_dev_deg, std_dev_deg)
    assert len(std_dev_deg) == 3, "std_dev_deg must be a float or a tuple/list of 3 floats"

    if rng is None:
        rng = np.random.default_rng()  # Do not fix the seed inside

    std_dev_rad = np.radians(std_dev_deg)
    rx, ry, rz = rng.normal(0.0, std_dev_rad)

    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    rot_y = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    rot_z = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    delta_R = rot_z @ rot_y @ rot_x
    return delta_R @ R


def random_translation_xyz(std_dev=0.01, rng=None):
    """
    Generate a random 3D translation vector from a normal distribution.

    Parameters:
        std_dev (float or array-like): Standard deviation(s) for each axis.
        rng (np.random.Generator): Optional random number generator for reproducibility.

    Returns:
        np.ndarray: A 3-element translation vector [tx, ty, tz].
    """
    if rng is None:
        rng = np.random.default_rng()

    std_dev = np.asarray(std_dev)
    if std_dev.size == 1:
        std_dev = np.full(3, std_dev)
    elif std_dev.size != 3:
        raise ValueError("std_dev must be a scalar or an array-like of 3 elements.")

    return rng.normal(loc=0.0, scale=std_dev, size=3)

    
def rotation_matrix_from_euler(rx_deg=0.0, ry_deg=0.0, rz_deg=0.0, order='xyz'):
    """
    Create a 3x3 rotation matrix from Euler angles (in degrees).

    Parameters
    ----------
    rx_deg, ry_deg, rz_deg : float
        Rotation angles around X, Y, Z axes (in degrees).
    order : str
        The order of rotation application, e.g., 'xyz', 'zyx', etc.
        Default is 'xyz' (i.e., R = Rz @ Ry @ Rx).

    Returns
    -------
    R : (3,3) ndarray
        Rotation matrix.
    """
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)]
    ])

    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [ 0,          1, 0         ],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,           0,          1]
    ])

    # Compose according to the desired order
    rotation_map = {'x': Rx, 'y': Ry, 'z': Rz}
    R = np.eye(3)
    for axis in order:
        R = rotation_map[axis] @ R

    return R
    
    

    
    
    
    
    
