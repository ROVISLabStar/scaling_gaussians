from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm
import numpy as np

def twist_to_transform(v):
    """Exponential map from twist vector to SE(3) transformation"""
    vx, vy, vz, wx, wy, wz = v
    v = np.array([vx, vy, vz])
    w = np.array([wx, wy, wz])
    wx_skew = np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])
    V = np.zeros((4, 4))
    V[:3, :3] = wx_skew
    V[:3, 3] = v
    return expm(V)

    
def opencv_to_opengl_transform():
    # Flip the Z axis (and optionally Y axis if needed)
    T = np.eye(4)
    T[1, 1] = -1  # Flip Y
    T[2, 2] = -1  # Flip Z
    return T
