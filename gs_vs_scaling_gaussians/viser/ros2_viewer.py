"""
ROS2-connected Viser Viewer for 3DGS Visual Servoing
=====================================================

Generic viewer — subscribes to ROS2 topics and displays:
- Full 3DGS Gaussian splat scene
- Drone with camera following the VS camera pose
- Camera frustums with live rendered images
- Camera trajectory
- Error display

No knowledge of start/goal — everything comes from the VS node.
Can stay running while multiple VS experiments are launched.

Usage:
    bash gs_vs_scaling_gaussians/scripts/launch_ros2_viewer.sh
    Then open http://localhost:8080

Author: Youssef ALJ (UM6P)
"""

import argparse
import os
import time
import threading
import numpy as np
import torch
import viser
import viser.transforms as tf
from viser.extras import ViserUrdf
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64, Float64MultiArray, String

from gs_vs.datasets.colmap import Parser
from gsplat.rendering import rasterization


def load_basic_cfg_fields(cfg_path):
    data = {}
    with open(cfg_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if k == "data_dir": data["data_dir"] = v
            elif k == "data_factor": data["data_factor"] = int(v)
            elif k == "normalize_world_space":
                data["normalize_world_space"] = v.lower() == "true"
    data.setdefault("data_factor", 1)
    data.setdefault("normalize_world_space", True)
    return data


class ROS2ViserViewer(Node):
    def __init__(self, server, ckpt_path, cfg_path, data_factor):
        super().__init__('viser_viewer')
        self.server = server
        self.device = "cuda"

        # Load GS scene
        self.get_logger().info("Loading GS scene...")
        cfg = load_basic_cfg_fields(cfg_path)
        parser = Parser(data_dir=cfg["data_dir"], factor=cfg["data_factor"],
                        normalize=cfg["normalize_world_space"], test_every=8)

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        splats = ckpt["splats"]

        self.camtoworlds = parser.camtoworlds
        n_views = len(self.camtoworlds)
        W_full, H_full = list(parser.imsize_dict.values())[0]
        K_colmap = list(parser.Ks_dict.values())[0]
        df = data_factor
        self.W, self.H = W_full // df, H_full // df
        fx = K_colmap[0, 0] / df
        self.fov_rad = 2 * np.arctan(self.W / (2 * fx))

        # Numpy data for splats
        means_np = splats["means"].cpu().numpy()
        quats_np = splats["quats"].cpu().numpy()
        scales_np = torch.exp(splats["scales"]).cpu().numpy()
        opacities_np = torch.sigmoid(splats["opacities"]).cpu().numpy()
        sh0_np = splats["sh0"].cpu().numpy()

        SH_C0 = 0.28209479177387814
        rgbs_np = np.clip(0.5 + SH_C0 * sh0_np[:, 0, :], 0.0, 1.0)
        Rs = tf.SO3(quats_np).as_matrix()
        covariances = np.einsum(
            "nij,njk,nlk->nil", Rs,
            np.eye(3)[None, :, :] * scales_np[:, None, :] ** 2, Rs)

        # Setup viser scene
        up_world = np.linalg.inv(self.camtoworlds[0])[:3, :3] @ np.array([0.0, -1.0, 0.0])
        up_world /= np.linalg.norm(up_world)
        server.scene.set_up_direction(tuple(up_world))

        # Add Gaussian splats
        self.get_logger().info(f"Loading {means_np.shape[0]} Gaussians...")
        server.scene.add_gaussian_splats(
            "/splats", centers=means_np, rgbs=rgbs_np,
            opacities=opacities_np[:, None], covariances=covariances)

        # Training cameras (sparse)
        for i in range(0, n_views, max(1, n_views // 20)):
            wxyz = tf.SO3.from_matrix(self.camtoworlds[i][:3, :3]).wxyz
            server.scene.add_camera_frustum(
                f"/cameras/view_{i}", fov=np.pi/3, aspect=self.W/self.H,
                scale=0.02, wxyz=wxyz, position=self.camtoworlds[i][:3, 3],
                color=(128, 128, 128))
            server.scene.add_label(f"/cameras/label_{i}", text=str(i),
                                   position=self.camtoworlds[i][:3, 3] + np.array([0, 0, 0.02]))

        # Drone URDF
        DRONE_URDF = Path(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ros2", "config", "drone.urdf"))
        self.urdf = ViserUrdf(server, DRONE_URDF, root_node_name="/drone", scale=1.0)

        # Current camera frustum (updated live from VS node)
        self.frustum_current = server.scene.add_camera_frustum(
            "/frustums/current", fov=self.fov_rad, aspect=self.W/self.H,
            scale=0.08, color=(80, 80, 255))

        # Goal and start frustums (set from VS node topics)
        self.frustum_goal = None
        self.frustum_start = None
        self.goal_pose_set = False
        self.start_pose_set = False
        self._desired_image = None   # cached until goal frustum is created
        self._start_image = None     # first current image = start view

        # Mode-specific trajectory colors
        self.mode_colors = {
            'original':  [255, 80, 80],    # red
            'inflated':  [80, 80, 255],    # blue
            'pgm_vs':    [255, 165, 0],    # orange
        }
        self.current_mode = 'unknown'
        self.traj_color = [80, 80, 255]  # default blue

        # State — per-mode trajectory segments
        # Each entry: (mode_name, color, [points])
        self.traj_segments = []
        self.iteration_count = 0

        # GUI
        with server.gui.add_folder("VS Status"):
            self.status_text = server.gui.add_text("Status", initial_value="Waiting for VS node...")
            self.error_text = server.gui.add_text("Error", initial_value="--")
            self.iter_text = server.gui.add_text("Iteration", initial_value="--")
        with server.gui.add_folder("Controls"):
            clear_traj = server.gui.add_button("Clear Trajectory")

        @clear_traj.on_click
        def _(_):
            for i, seg in enumerate(self.traj_segments):
                try:
                    server.scene.remove(f"/trajectory/{i}")
                except Exception:
                    pass
            self.traj_segments.clear()

        # ROS2 subscribers
        self.create_subscription(
            Float64MultiArray, '/vs/camera_c2w', self.pose_cb, 10)
        self.create_subscription(
            Image, '/vs/current_image', self.image_cb, 1)
        self.create_subscription(
            Image, '/vs/desired_image', self.desired_image_cb, 1)
        self.create_subscription(
            Float64, '/vs/photometric_error', self.error_cb, 10)
        self.create_subscription(
            Float64MultiArray, '/vs/goal_c2w', self.goal_pose_cb, 10)
        self.create_subscription(
            Float64MultiArray, '/vs/start_c2w', self.start_pose_cb, 10)
        self.create_subscription(
            String, '/vs/mode', self.mode_cb, 10)

        self.get_logger().info("Viewer ready. Waiting for VS node...")

    def _update_drone_pose(self, c2w):
        """Position drone level at camera position, facing forward."""
        pos = c2w[:3, 3]
        cam_forward = c2w[:3, 2]

        drone_forward = cam_forward.copy()
        drone_forward[2] = 0
        norm = np.linalg.norm(drone_forward)
        if norm > 1e-6:
            drone_forward /= norm
        else:
            drone_forward = np.array([1, 0, 0])

        drone_z = np.array([0, 0, 1.0])
        drone_x = drone_forward
        drone_y = np.cross(drone_z, drone_x)
        drone_y /= np.linalg.norm(drone_y) + 1e-8

        R_drone = np.column_stack([drone_x, drone_y, drone_z])
        wxyz = tf.SO3.from_matrix(R_drone).wxyz
        self.server.scene.add_frame("/drone", position=pos, wxyz=wxyz,
                                    axes_length=0.0, axes_radius=0.0)

    def pose_cb(self, msg):
        """Update drone and frustum from actual VS camera pose."""
        try:
            c2w = np.array(msg.data).reshape(4, 4)
            self._update_drone_pose(c2w)

            self.frustum_current.wxyz = tf.SO3.from_matrix(c2w[:3, :3]).wxyz
            self.frustum_current.position = c2w[:3, 3]

            # Append to current segment
            if not self.traj_segments or self.traj_segments[-1][0] != self.current_mode:
                self.traj_segments.append((self.current_mode, self.traj_color, []))
            self.traj_segments[-1][2].append(c2w[:3, 3].copy())

            seg_idx = len(self.traj_segments) - 1
            seg_pts = self.traj_segments[-1][2]
            if len(seg_pts) > 1 and len(seg_pts) % 3 == 0:
                pts = np.array(seg_pts)
                color = self.traj_segments[-1][1]
                self.server.scene.add_point_cloud(
                    f"/trajectory/{seg_idx}", points=pts,
                    colors=np.full((len(pts), 3), color, dtype=np.uint8),
                    point_size=0.005)

            self.status_text.value = "Running"
        except Exception:
            pass

    def image_cb(self, msg):
        """Update current frustum image. First image is cached for start frustum."""
        try:
            h, w = msg.height, msg.width
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            img_ds = (img[::2, ::2].astype(np.float32) / 255.0).clip(0, 1)
            self.frustum_current.image = img_ds
            self.iteration_count += 1
            self.iter_text.value = str(self.iteration_count)
            # Cache first image as start view
            if self._start_image is None:
                self._start_image = img_ds
                if self.frustum_start is not None:
                    self.frustum_start.image = self._start_image
        except Exception:
            pass

    def desired_image_cb(self, msg):
        """Cache desired image and attach to goal frustum when available."""
        if self._desired_image is not None:
            return  # already set
        try:
            h, w = msg.height, msg.width
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            self._desired_image = (img[::2, ::2].astype(np.float32) / 255.0).clip(0, 1)
            self.get_logger().info("Received desired image")
            if self.frustum_goal is not None:
                self.frustum_goal.image = self._desired_image
        except Exception:
            pass

    def goal_pose_cb(self, msg):
        """Create/update green goal frustum from VS node."""
        try:
            c2w = np.array(msg.data).reshape(4, 4)
            wxyz = tf.SO3.from_matrix(c2w[:3, :3]).wxyz
            pos = c2w[:3, 3]
            if self.frustum_goal is None:
                self.frustum_goal = self.server.scene.add_camera_frustum(
                    "/frustums/goal", fov=self.fov_rad, aspect=self.W/self.H,
                    scale=0.08, color=(0, 200, 0), wxyz=wxyz, position=pos)
                self.goal_pose_set = True
                self.get_logger().info("Goal frustum set")
                self.server.scene.add_label("/frustums/goal_label", text="Desired",
                                            position=pos + np.array([0, 0, 0.03]))
                if self._desired_image is not None:
                    self.frustum_goal.image = self._desired_image
            else:
                self.frustum_goal.wxyz = wxyz
                self.frustum_goal.position = pos
        except Exception:
            pass

    def start_pose_cb(self, msg):
        """Create/update red start frustum from VS node."""
        try:
            c2w = np.array(msg.data).reshape(4, 4)
            wxyz = tf.SO3.from_matrix(c2w[:3, :3]).wxyz
            pos = c2w[:3, 3]
            if self.frustum_start is None:
                self.frustum_start = self.server.scene.add_camera_frustum(
                    "/frustums/start", fov=self.fov_rad, aspect=self.W/self.H,
                    scale=0.08, color=(200, 0, 0), wxyz=wxyz, position=pos)
                self.start_pose_set = True
                self.get_logger().info("Start frustum set")
                self.server.scene.add_label("/frustums/start_label", text="Initial",
                                            position=pos + np.array([0, 0, 0.03]))
                if self._start_image is not None:
                    self.frustum_start.image = self._start_image
            else:
                self.frustum_start.wxyz = wxyz
                self.frustum_start.position = pos
        except Exception:
            pass

    def mode_cb(self, msg):
        """Update trajectory color when VS mode changes."""
        mode = msg.data
        if mode != self.current_mode:
            self.current_mode = mode
            self.traj_color = self.mode_colors.get(mode, [80, 80, 255])
            self.status_text.value = f"Running ({mode})"
            self.get_logger().info(f"Mode: {mode}")

    def error_cb(self, msg):
        self.error_text.value = f"{msg.data:.0f}"


def main():
    p = argparse.ArgumentParser(description="ROS2 Viser Viewer (generic)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--data_factor", type=int, default=8)
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_layout="collapsible", control_width="large")
    print(f"[Viser] http://localhost:{args.port}")

    rclpy.init()
    node = ROS2ViserViewer(server, args.ckpt, args.cfg, args.data_factor)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("[Viewer] Running. Start VS node in another terminal.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
