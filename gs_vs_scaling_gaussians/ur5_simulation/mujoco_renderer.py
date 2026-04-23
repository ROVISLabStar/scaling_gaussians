"""
MuJoCo-based UR10e external view renderer with 3DGS scene compositing
and camera frustum visualization.
"""

import os
import numpy as np
import mujoco
import torch
import cv2
from gsplat.rendering import rasterization


MODEL_DIR = os.path.join(os.path.dirname(__file__),
                         "mujoco_models", "mujoco_menagerie", "universal_robots_ur10e")


class MuJoCoUR10eRenderer:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

        # Load model
        xml_path = os.path.join(MODEL_DIR, "scene.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.vis.global_.offwidth = max(width, 1280)
        self.model.vis.global_.offheight = max(height, 960)
        self.data = mujoco.MjData(self.model)

        # Offscreen renderer
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        # Camera
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance = 2.0
        self.cam.azimuth = 135
        self.cam.elevation = -25
        self.cam.lookat[:] = [0.3, 0, 0.4]

        # GS scene data
        self.gs_data = None

        # eMc transform for frustum drawing
        self.eMc = None

        # Camera frustum parameters
        self.frustum_length = 0.08   # frustum length in meters
        self.frustum_fov = 60        # degrees

    def set_camera(self, distance=2.0, azimuth=135, elevation=-25,
                   lookat=(0.3, 0, 0.4)):
        self.cam.distance = distance
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        self.cam.lookat[:] = lookat

    def set_eMc(self, eMc):
        """Set the end-effector to camera transform for frustum visualization."""
        self.eMc = eMc

    def init_gs_scene(self, means, quats, scales, opacities, colors,
                      sh_degree, bMo, device="cuda"):
        self.gs_data = {
            "means": means, "quats": quats, "scales": scales,
            "opacities": opacities, "colors": colors,
            "sh_degree": sh_degree, "bMo": bMo, "device": device,
        }

    def _get_external_view_matrix(self):
        """Get 4x4 view matrix for the MuJoCo external camera (camera-to-base)."""
        dist = self.cam.distance
        az = np.radians(self.cam.azimuth)
        el = np.radians(self.cam.elevation)
        lookat = np.array(self.cam.lookat)

        x = dist * np.cos(el) * np.cos(az)
        y = dist * np.cos(el) * np.sin(az)
        z = dist * np.sin(el)
        cam_pos = lookat + np.array([x, y, z])

        forward = lookat - cam_pos
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)

        # OpenGL convention: camera looks along -Z
        R = np.array([right, up, -forward])
        t = -R @ cam_pos

        cMb = np.eye(4)
        cMb[:3, :3] = R
        cMb[:3, 3] = t
        return cMb

    def _get_ee_pose(self, q):
        """Get end-effector pose in base frame."""
        for i in range(min(6, self.model.nq)):
            self.data.qpos[i] = q[i]
        mujoco.mj_forward(self.model, self.data)

        # Get end-effector site/body position
        # The last body is the end-effector
        ee_id = self.model.nbody - 1
        ee_pos = self.data.xpos[ee_id].copy()
        ee_rot = self.data.xmat[ee_id].reshape(3, 3).copy()

        bMe = np.eye(4)
        bMe[:3, :3] = ee_rot
        bMe[:3, 3] = ee_pos
        return bMe

    def _project_point(self, point_base, cMb, fov_deg):
        """Project a 3D point (base frame) to 2D pixel coordinates.
        Camera looks along -Z (OpenGL convention), so points in front have z < 0."""
        p_cam = cMb[:3, :3] @ point_base + cMb[:3, 3]
        # OpenGL convention: camera looks along -Z, so in-front points have z < 0
        if p_cam[2] >= 0:
            return None  # point behind camera
        z = -p_cam[2]  # positive depth
        fx = self.width / (2 * np.tan(np.radians(fov_deg / 2)))
        fy = fx
        u = fx * (-p_cam[0]) / z + self.width / 2  # flip x for screen coords
        v = fx * (-p_cam[1]) / z + self.height / 2  # flip y for screen coords
        return (int(u), int(v))

    def _draw_frustum(self, frame, q):
        """Draw camera frustum on the rendered frame at the end-effector."""
        if self.eMc is None:
            return frame

        bMe = self._get_ee_pose(q)
        bMc = bMe @ self.eMc  # camera pose in base frame

        cam_origin = bMc[:3, 3]
        cam_z = bMc[:3, 2]  # camera optical axis (looking direction)
        cam_x = bMc[:3, 0]  # camera right
        cam_y = bMc[:3, 1]  # camera up (or down in OpenCV)

        L = self.frustum_length
        half_w = L * np.tan(np.radians(self.frustum_fov / 2))

        # Frustum corners at distance L along optical axis
        center_far = cam_origin + cam_z * L
        corners = [
            center_far + cam_x * half_w + cam_y * half_w,   # top-right
            center_far - cam_x * half_w + cam_y * half_w,   # top-left
            center_far - cam_x * half_w - cam_y * half_w,   # bottom-left
            center_far + cam_x * half_w - cam_y * half_w,   # bottom-right
        ]

        cMb = self._get_external_view_matrix()
        fov = 60  # external camera FOV

        # Project origin and corners
        origin_2d = self._project_point(cam_origin, cMb, fov)
        corner_2ds = [self._project_point(c, cMb, fov) for c in corners]

        if origin_2d is None or any(c is None for c in corner_2ds):
            return frame

        frame = frame.copy()

        # Draw frustum edges (origin to each corner)
        color_frustum = (0, 255, 0)  # green
        for c2d in corner_2ds:
            cv2.line(frame, origin_2d, c2d, color_frustum, 2)

        # Draw far plane rectangle
        for i in range(4):
            cv2.line(frame, corner_2ds[i], corner_2ds[(i + 1) % 4], color_frustum, 2)

        # Draw camera center dot
        cv2.circle(frame, origin_2d, 5, (0, 0, 255), -1)  # red dot

        # Draw optical axis
        axis_end = cam_origin + cam_z * L * 1.3
        axis_2d = self._project_point(axis_end, cMb, fov)
        if axis_2d is not None:
            cv2.arrowedLine(frame, origin_2d, axis_2d, (255, 0, 0), 2, tipLength=0.3)

        return frame

    def _draw_goal_frustum(self, frame, bMc_goal):
        """Draw the goal camera frustum in a different color."""
        cam_origin = bMc_goal[:3, 3]
        cam_z = bMc_goal[:3, 2]
        cam_x = bMc_goal[:3, 0]
        cam_y = bMc_goal[:3, 1]

        L = self.frustum_length
        half_w = L * np.tan(np.radians(self.frustum_fov / 2))

        center_far = cam_origin + cam_z * L
        corners = [
            center_far + cam_x * half_w + cam_y * half_w,
            center_far - cam_x * half_w + cam_y * half_w,
            center_far - cam_x * half_w - cam_y * half_w,
            center_far + cam_x * half_w - cam_y * half_w,
        ]

        cMb = self._get_external_view_matrix()
        fov = 60

        origin_2d = self._project_point(cam_origin, cMb, fov)
        corner_2ds = [self._project_point(c, cMb, fov) for c in corners]

        if origin_2d is None or any(c is None for c in corner_2ds):
            return frame

        frame = frame.copy()
        color = (0, 200, 0)  # dark green for goal

        for c2d in corner_2ds:
            cv2.line(frame, origin_2d, c2d, color, 1)
        for i in range(4):
            cv2.line(frame, corner_2ds[i], corner_2ds[(i + 1) % 4], color, 1)
        cv2.circle(frame, origin_2d, 4, color, -1)

        return frame

    @torch.no_grad()
    def _render_gs_background(self):
        """Render the GS scene from the external camera viewpoint."""
        if self.gs_data is None:
            return None

        d = self.gs_data
        cMb = self._get_external_view_matrix()
        cMo = cMb @ d["bMo"]

        fov_deg = 60
        fx = self.width / (2 * np.tan(np.radians(fov_deg / 2)))
        fy = fx
        cx, cy = self.width / 2.0, self.height / 2.0
        K_ext = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

        viewmat = torch.from_numpy(cMo).float().to(d["device"])[None]
        Ks = torch.from_numpy(K_ext).float().to(d["device"])[None]

        renders, _, _ = rasterization(
            means=d["means"], quats=d["quats"], scales=d["scales"],
            opacities=d["opacities"], colors=d["colors"],
            sh_degree=d["sh_degree"], viewmats=viewmat, Ks=Ks,
            width=self.width, height=self.height, packed=True,
            render_mode="RGB+ED", camera_model="pinhole",
        )
        rgb = torch.clamp(renders[0, ..., :3], 0.0, 1.0)
        return (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    def render(self, q):
        """Render robot only. Returns (H, W, 3) uint8 RGB."""
        for i in range(min(6, self.model.nq)):
            self.data.qpos[i] = q[i]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.cam)
        return self.renderer.render()

    def render_composite(self, q, bMc_goal=None):
        """
        Render robot composited onto GS scene with camera frustums.

        Args:
            q: (6,) joint angles
            bMc_goal: (4,4) goal camera pose in base frame (for goal frustum)

        Returns:
            frame: (H, W, 3) uint8 RGB
        """
        # Set joints and forward kinematics
        for i in range(min(6, self.model.nq)):
            self.data.qpos[i] = q[i]
        mujoco.mj_forward(self.model, self.data)

        # Render robot RGB
        self.renderer.update_scene(self.data, self.cam)
        robot_rgb = self.renderer.render().copy()

        # Get robot-only depth mask
        scene_opt = mujoco.MjvOption()
        scene_opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = False
        self.renderer.update_scene(self.data, self.cam, scene_option=scene_opt)
        self.renderer.enable_depth_rendering()
        depth_no_floor = self.renderer.render()
        self.renderer.disable_depth_rendering()

        robot_mask = depth_no_floor < (0.98 * depth_no_floor.max())

        # Restore normal scene
        self.renderer.update_scene(self.data, self.cam)

        # Render GS background
        gs_bg = self._render_gs_background()
        if gs_bg is None:
            composite = robot_rgb
        else:
            composite = gs_bg.copy()
            composite[robot_mask] = robot_rgb[robot_mask]

        # Convert to BGR for cv2 drawing, then back
        composite_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

        # Draw current camera frustum
        composite_bgr = self._draw_frustum(composite_bgr, q)

        # Draw goal camera frustum
        if bMc_goal is not None:
            composite_bgr = self._draw_goal_frustum(composite_bgr, bMc_goal)

        return cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        self.renderer.close()
