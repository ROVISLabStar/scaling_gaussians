"""
UR Robot Kinematics
====================

Forward kinematics, geometric Jacobian, and velocity control
for Universal Robots manipulators (UR5e, UR10e).

DH parameters from UR datasheets (modified DH convention, Craig).

Coordinate frames:
  - Object frame (o): the GS scene world frame
  - Base frame (b): robot base, placed via bMo transform
  - Tool frame (e): end-effector flange
  - Camera frame (c): mounted on tool with fixed eMc

Author: Youssef ALJ (UM6P)
"""

import numpy as np


# ============================================================
# DH Parameters
# ============================================================
UR_MODELS = {
    "ur5e": {
        "a":     [0.0,     -0.42500, -0.39225, 0.0,      0.0,      0.0],
        "d":     [0.1625,   0.0,      0.0,     0.1333,   0.0997,   0.0996],
        "alpha": [np.pi/2,  0.0,      0.0,     np.pi/2, -np.pi/2,  0.0],
        "reach": 0.85,
    },
    "ur10e": {
        "a":     [0.0,     -0.6127,  -0.57155, 0.0,      0.0,      0.0],
        "d":     [0.1807,   0.0,      0.0,     0.17415,  0.11985,  0.11655],
        "alpha": [np.pi/2,  0.0,      0.0,     np.pi/2, -np.pi/2,  0.0],
        "reach": 1.30,
    },
}

# Joint limits (same for all UR models)
UR_JOINT_LIMITS = {
    "lower": np.array([-2*np.pi] * 6),
    "upper": np.array([2*np.pi] * 6),
}
UR_MAX_JOINT_VEL = np.array([np.pi] * 6)


def dh_matrix(a, d, alpha, theta):
    """4x4 homogeneous transform for one DH link (modified convention)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,     -st,     0,      a],
        [st*ca,   ct*ca, -sa,   -d*sa],
        [st*sa,   ct*sa,  ca,    d*ca],
        [0,       0,      0,     1],
    ])


def forward_kinematics(q, model="ur10e", return_all=False):
    """
    Forward kinematics.

    Args:
        q: (6,) joint angles in radians
        model: "ur5e" or "ur10e"
        return_all: if True, return all intermediate transforms

    Returns:
        T: (4, 4) base-to-end-effector (bMe)
    """
    params = UR_MODELS[model]
    a, d, alpha = params["a"], params["d"], params["alpha"]

    T = np.eye(4)
    transforms = [T.copy()]
    for i in range(6):
        T = T @ dh_matrix(a[i], d[i], alpha[i], q[i])
        transforms.append(T.copy())

    return (T, transforms) if return_all else T


def geometric_jacobian(q, model="ur10e"):
    """6x6 geometric Jacobian in the base frame."""
    _, transforms = forward_kinematics(q, model, return_all=True)
    p_ee = transforms[6][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z_i = transforms[i][:3, 2]
        p_i = transforms[i][:3, 3]
        J[:3, i] = np.cross(z_i, p_ee - p_i)
        J[3:, i] = z_i
    return J


def default_eMc():
    """
    Eye-in-hand camera mount.
    Camera looks along tool -Z (OpenCV convention: camera Z = forward).
    """
    eMc = np.eye(4)
    eMc[:3, :3] = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ])
    return eMc


# ============================================================
# Robot simulator
# ============================================================
class UR_Simulator:
    """
    Simulates a UR robot with eye-in-hand camera in a GS scene.

    The robot base is placed so that a "home" joint configuration
    puts the camera at the desired goal pose. This ensures the
    entire VS workspace is reachable.
    """

    def __init__(self, model="ur10e", eMc=None, dt=0.04):
        self.model = model
        self.eMc = eMc if eMc is not None else default_eMc()
        self.dt = dt
        self.q_home = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
        self.q = self.q_home.copy()
        self.bMo = np.eye(4)  # base-to-object (GS world)

    def place_at_pose(self, target_c2w):
        """
        Place the robot base so that q_home puts the camera at target_c2w.
        """
        bMe_home = forward_kinematics(self.q_home, self.model)
        bMc_home = bMe_home @ self.eMc
        oMb = target_c2w @ np.linalg.inv(bMc_home)
        self.bMo = np.linalg.inv(oMb)
        self.q = self.q_home.copy()

    def place_base_fixed(self, base_position, base_orientation=None):
        """
        Place the robot base at a fixed position in the GS scene.

        The robot base Z-axis points up. The base is positioned so it
        appears physically grounded in the scene.

        Args:
            base_position: (3,) position in GS object frame
            base_orientation: (3,3) rotation matrix for base, or None for identity
        """
        oMb = np.eye(4)
        oMb[:3, 3] = base_position
        if base_orientation is not None:
            oMb[:3, :3] = base_orientation
        self.bMo = np.linalg.inv(oMb)
        self.q = self.q_home.copy()

    def move_to_pose(self, target_c2w, max_iter=5000, gain=2.0):
        """
        Move the robot to put the camera at target_c2w using iterative IK.

        Args:
            target_c2w: (4,4) desired camera-to-world in GS object frame
            max_iter: max IK iterations
            gain: velocity gain

        Returns:
            success: bool
            pose_error: (t_err, r_err) final pose error
        """
        from scipy.spatial.transform import Rotation as Rot

        for it in range(max_iter):
            _, c2w_cur = self.get_camera_pose()
            cMc_target = np.linalg.inv(c2w_cur) @ target_c2w
            dt_err = cMc_target[:3, 3]
            dR_err = Rot.from_matrix(cMc_target[:3, :3]).as_rotvec()

            t_err = np.linalg.norm(dt_err)
            r_err = np.degrees(np.linalg.norm(dR_err))

            if t_err < 1e-4 and r_err < 0.1:
                return True, (t_err, r_err)

            v_cam = gain * np.concatenate([dt_err, dR_err])
            # Clamp velocity
            vt = np.linalg.norm(v_cam[:3])
            vr = np.linalg.norm(v_cam[3:])
            if vt > 1.0: v_cam[:3] *= 1.0 / vt
            if vr > 1.0: v_cam[3:] *= 1.0 / vr

            self.set_camera_velocity(v_cam)

        _, c2w_final = self.get_camera_pose()
        cMc = np.linalg.inv(c2w_final) @ target_c2w
        t_err = np.linalg.norm(cMc[:3, 3])
        r_err = np.degrees(np.linalg.norm(Rot.from_matrix(cMc[:3, :3]).as_rotvec()))
        return False, (t_err, r_err)

    def get_camera_pose(self):
        """
        Returns:
            cMo: camera-to-object (for gsplat rendering, = view matrix)
            c2w: camera-to-world in GS object frame (for pose comparison)
        """
        bMe = forward_kinematics(self.q, self.model)
        bMc = bMe @ self.eMc
        # oMc = oMb @ bMc = inv(bMo) @ bMc
        oMc = np.linalg.inv(self.bMo) @ bMc
        cMo = np.linalg.inv(oMc)  # view matrix for rendering
        return cMo, oMc  # oMc = c2w

    def set_camera_velocity(self, v_camera):
        """
        Apply a camera-frame velocity by converting to joint velocities.

        Args:
            v_camera: (6,) velocity in camera frame [vx, vy, vz, wx, wy, wz]
        """
        # Camera Jacobian: maps dq to v_camera
        bMe = forward_kinematics(self.q, self.model)
        bMc = bMe @ self.eMc
        cRb = bMc[:3, :3].T

        # Twist transform base->camera
        cWb = np.zeros((6, 6))
        cWb[:3, :3] = cRb
        cWb[3:, 3:] = cRb

        J_base = geometric_jacobian(self.q, self.model)
        J_cam = cWb @ J_base

        # Damped least squares
        damping = 0.01
        JJt = J_cam @ J_cam.T + damping**2 * np.eye(6)
        dq = J_cam.T @ np.linalg.solve(JJt, v_camera)

        # Clamp joint velocities
        scale = np.max(np.abs(dq) / UR_MAX_JOINT_VEL)
        if scale > 1.0:
            dq /= scale

        # Integrate
        self.q = self.q + dq * self.dt

    def check_joint_limits(self):
        return np.all(self.q >= UR_JOINT_LIMITS["lower"]) and \
               np.all(self.q <= UR_JOINT_LIMITS["upper"])

    def get_joints_deg(self):
        return np.degrees(self.q).astype(int)
