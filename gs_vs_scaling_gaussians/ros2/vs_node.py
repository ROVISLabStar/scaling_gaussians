#!/usr/bin/python3
"""
ROS2 Visual Servoing Node for UR10e with 3DGS
===============================================

1. Moves robot to initial configuration via position controller
2. Switches to velocity controller
3. Runs VS loop: gsplat render → error → velocity → joint velocity → publish

Topics:
  Sub: /joint_states
  Pub: /forward_velocity_controller/commands (joint velocities)
  Pub: /forward_position_controller/commands (initial positioning)
  Pub: /vs/current_image, /vs/desired_image, /vs/diff_image
"""

import sys
import os
import numpy as np
import torch
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, PointCloud2, PointField
from std_msgs.msg import Float64MultiArray, Float64, Header, String
from cv_bridge import CvBridge
import tf2_ros
from scipy.spatial.transform import Rotation as Rot
import struct

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

from gs_vs.simulators.RobotSimulator import SimulatorCamera
from gs_vs.simulators.CameraParameters import CameraParameters
from gs_vs.features.factory import create_feature
from gs_vs.datasets.colmap import Parser
from gsplat.rendering import rasterization
from gs_vs_scaling_gaussians.ur5_simulation.ur5_kinematics import (
    forward_kinematics, geometric_jacobian, default_eMc,
)


class VSNode(Node):
    def __init__(self):
        super().__init__('vs_node')

        # Parameters
        self.declare_parameter('ckpt', '')
        self.declare_parameter('cfg', '')
        self.declare_parameter('data_factor', 8)
        self.declare_parameter('goal_idx', 10)
        self.declare_parameter('start_idx', 14)
        self.declare_parameter('mode', 'inflated')
        self.declare_parameter('scale_factor', 1.8)
        self.declare_parameter('gain', 10.0)
        self.declare_parameter('mu', 0.01)
        self.declare_parameter('convergence_threshold', 10.0)
        self.declare_parameter('max_iter', 2000)
        self.declare_parameter('rate', 10.0)
        self.declare_parameter('robot_model', 'ur10e')
        # PGM-VS parameters
        self.declare_parameter('pgm_lambda_init', 5.0)
        self.declare_parameter('pgm_lambda_final', 1.0)
        self.declare_parameter('pgm_gain', 10.0)

        ckpt_path = self.get_parameter('ckpt').value
        cfg_path = self.get_parameter('cfg').value
        data_factor = self.get_parameter('data_factor').value
        self.goal_idx = self.get_parameter('goal_idx').value
        self.start_idx = self.get_parameter('start_idx').value
        self.mode = self.get_parameter('mode').value
        self.scale_factor = self.get_parameter('scale_factor').value
        self.gain = self.get_parameter('gain').value
        self.mu = self.get_parameter('mu').value
        self.convergence_threshold = self.get_parameter('convergence_threshold').value
        self.max_iter = self.get_parameter('max_iter').value
        self.pgm_lambda_init = self.get_parameter('pgm_lambda_init').value
        self.pgm_lambda_final = self.get_parameter('pgm_lambda_final').value
        self.pgm_gain = self.get_parameter('pgm_gain').value
        rate = self.get_parameter('rate').value
        self.robot_model = self.get_parameter('robot_model').value

        self.device = 'cuda'
        self.eMc = default_eMc()
        self.bridge = CvBridge()

        # q_init: read from first joint_states message (set in joint_cb)
        # For fake hardware, UR10e default is approx [-1.57, 0, -1.57, 0, 0, 0]
        self.q_init = None  # Will be set from first joint state

        # Load GS scene
        self.get_logger().info(f'Loading GS scene...')
        cfg = self._load_cfg(cfg_path)
        parser = Parser(data_dir=cfg["data_dir"], factor=cfg["data_factor"],
                        normalize=cfg["normalize_world_space"], test_every=8)

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        splats = ckpt["splats"]
        self.means = splats["means"].to(self.device)
        self.quats = splats["quats"].to(self.device)
        self.scales_original = torch.exp(splats["scales"]).to(self.device)
        self.opacities = torch.sigmoid(splats["opacities"]).to(self.device)
        self.colors = torch.cat([splats["sh0"], splats["shN"]], dim=1).to(self.device)
        self.sh_degree = int(np.sqrt(self.colors.shape[1]) - 1)

        self.camtoworlds = parser.camtoworlds
        W_full, H_full = list(parser.imsize_dict.values())[0]
        K_colmap = list(parser.Ks_dict.values())[0]
        df = data_factor
        self.W, self.H = W_full // df, H_full // df
        fx, fy = K_colmap[0, 0] / df, K_colmap[1, 1] / df
        cx, cy = K_colmap[0, 2] / df, K_colmap[1, 2] / df
        self.K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        self.cam_params = CameraParameters(px=fx, py=fy, u0=cx, v0=cy)

        # Scale
        sf = self.scale_factor if self.mode == 'inflated' else 1.0
        self.scales_cur = self.scales_original * sf

        # Robot base: will be computed once we receive the first joint state
        self.c2w_start = self.camtoworlds[self.start_idx]
        self.bMo = None  # Set in loop_cb when q_init is known

        # Desired features
        self.is_pgm = (self.mode == 'pgm_vs')
        c2w_goal = self.camtoworlds[self.goal_idx]
        cMo_goal = np.linalg.inv(c2w_goal)
        _, gray_des, depth_des = self._render(cMo_goal)
        self.gray_des = gray_des
        self.depth_des = depth_des

        if self.is_pgm:
            sys.path.insert(0, os.path.join(PROJECT_DIR, 'gs_vs_pgm_vs'))
            from features.FeaturePGM import FeaturePGM
            self.FeaturePGM = FeaturePGM
            # Build PGM lambda levels
            self.pgm_lambdas = []
            lam = self.pgm_lambda_init
            while lam > 2.0 * self.pgm_lambda_final:
                self.pgm_lambdas.append(lam)
                lam *= 0.5
            self.pgm_lambdas.append(self.pgm_lambda_final)
            self.pgm_level = 0
            self.pgm_stall = 0
            self.pgm_prev_mse = float('inf')
            cur_lam = self.pgm_lambdas[0]
            self.s_star = FeaturePGM(lambda_g=cur_lam, border=10, device=self.device)
        else:
            self.s_star = create_feature('pinhole', device=self.device, border=10)

        self.s_star.init(self.H, self.W)
        self.s_star.setCameraParameters(self.cam_params)
        self.s_star.buildFrom(gray_des, depth_des)

        # Desired image for publishing
        rgb_des, _, _ = self._render(cMo_goal)
        self.rgb_des_np = rgb_des.cpu().numpy()

        self.get_logger().info(f'Scene: {len(self.camtoworlds)} views, {self.W}x{self.H}')
        self.get_logger().info(f'Pair: {self.start_idx} -> {self.goal_idx}')
        self.get_logger().info(f'Mode: {self.mode}, sf={sf}')

        # Joint names (UR convention)
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
        ]

        # State
        self.current_q = None
        self.iteration = 0
        self.running = True

        # ROS2 interfaces
        self.sub_joints = self.create_subscription(
            JointState, '/joint_states', self.joint_cb, 10)
        # TF listener — get actual EE pose from the driver's own FK
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_vel = self.create_publisher(
            Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.pub_image = self.create_publisher(Image, '/vs/current_image', 10)
        self.pub_desired = self.create_publisher(Image, '/vs/desired_image', 10)
        self.pub_diff = self.create_publisher(Image, '/vs/diff_image', 10)
        self.pub_pc = self.create_publisher(PointCloud2, '/vs/scene_pointcloud', 1)
        self.pub_error = self.create_publisher(Float64, '/vs/photometric_error', 10)
        self.pub_pose_t = self.create_publisher(Float64, '/vs/pose_error_t', 10)
        self.pub_pose_r = self.create_publisher(Float64, '/vs/pose_error_r', 10)
        self.pub_camera_pose = self.create_publisher(Float64MultiArray, '/vs/camera_c2w', 10)
        self.pub_goal_pose = self.create_publisher(Float64MultiArray, '/vs/goal_c2w', 10)
        self.pub_start_pose = self.create_publisher(Float64MultiArray, '/vs/start_c2w', 10)
        self.pub_mode = self.create_publisher(String, '/vs/mode', 10)

        # Publish start and goal poses (periodically so viewer can pick them up)
        self._c2w_goal = np.linalg.inv(np.linalg.inv(self.camtoworlds[self.goal_idx]))
        self._c2w_start = self.camtoworlds[self.start_idx].copy()
        self.pose_info_timer = self.create_timer(1.0, self._publish_pose_info)

        # Publish GS scene as PointCloud2 (once, latched via transient_local)
        self._publish_gs_pointcloud()

        # Timer to re-publish point cloud periodically (RViz needs it)
        self.pc_timer = self.create_timer(10.0, self._publish_gs_pointcloud)

        # Timer
        self.timer = self.create_timer(1.0 / rate, self.loop_cb)
        self.get_logger().info('VS node ready. Waiting for joint states...')

    def _publish_gs_pointcloud(self):
        """Publish 3DGS scene as colored PointCloud2 for RViz."""
        means_np = self.means.cpu().numpy()
        # Get colors from SH0
        sh0 = torch.cat([self.colors[:, :1, :]], dim=1)  # first SH band
        SH_C0 = 0.28209479177387814
        rgbs = np.clip(0.5 + SH_C0 * sh0[:, 0, :].cpu().numpy(), 0.0, 1.0)

        # Subsample for performance
        n = means_np.shape[0]
        max_pts = 30000
        if n > max_pts:
            idx = np.random.choice(n, max_pts, replace=False)
            pts = means_np[idx]
            colors = rgbs[idx]
        else:
            pts = means_np
            colors = rgbs

        # Transform points from GS object frame to robot base frame
        # p_base = bMo @ p_object (bMo maps object→base)
        # But bMo might not be computed yet, so use identity first time
        if self.bMo is not None:
            oMb = np.linalg.inv(self.bMo)
            # p_object = oMb @ p_base → p_base = inv(oMb) @ p_object = bMo @ p_object
            pts_base = (self.bMo[:3, :3] @ pts.T + self.bMo[:3, 3:4]).T
        else:
            pts_base = pts

        # Build PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        # Pack data
        point_data = []
        for i in range(len(pts_base)):
            r, g, b = int(colors[i, 0] * 255), int(colors[i, 1] * 255), int(colors[i, 2] * 255)
            rgb_packed = (r << 16) | (g << 8) | b
            point_data.append(struct.pack('fffI', pts_base[i, 0], pts_base[i, 1], pts_base[i, 2], rgb_packed))

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(pts_base)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * len(pts_base)
        msg.data = b''.join(point_data)
        msg.is_dense = True

        self.pub_pc.publish(msg)

    def _publish_pose_info(self):
        """Publish start and goal poses periodically."""
        goal_msg = Float64MultiArray()
        goal_msg.data = self._c2w_goal.flatten().tolist()
        self.pub_goal_pose.publish(goal_msg)
        start_msg = Float64MultiArray()
        start_msg.data = self._c2w_start.flatten().tolist()
        self.pub_start_pose.publish(start_msg)
        mode_msg = String()
        mode_msg.data = self.mode
        self.pub_mode.publish(mode_msg)

    def _load_cfg(self, path):
        data = {}
        with open(path) as f:
            for line in f:
                if ":" not in line: continue
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if k == "data_dir": data["data_dir"] = v
                elif k == "data_factor": data["data_factor"] = int(v)
                elif k == "normalize_world_space": data["normalize_world_space"] = v.lower() == "true"
        data.setdefault("data_factor", 1)
        data.setdefault("normalize_world_space", True)
        return data

    @torch.no_grad()
    def _render(self, cMo):
        return self._render_with_scales(cMo, self.scales_cur)

    @torch.no_grad()
    def _render_with_scales(self, cMo, scales):
        viewmat = torch.from_numpy(cMo).float().to(self.device)[None]
        Ks = torch.from_numpy(self.K_np).float().to(self.device)[None]
        renders, _, _ = rasterization(
            means=self.means, quats=self.quats, scales=scales,
            opacities=self.opacities, colors=self.colors,
            sh_degree=self.sh_degree, viewmats=viewmat, Ks=Ks,
            width=self.W, height=self.H, packed=True,
            render_mode="RGB+ED", camera_model="pinhole")
        rgb = torch.clamp(renders[0, ..., :3], 0.0, 1.0)
        depth = renders[0, ..., 3]
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return rgb, gray, depth

    def _get_camera_pose(self, q):
        """Get camera pose using TF (driver's own FK) or fallback to our FK."""
        try:
            t = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            # Convert TF to 4x4 matrix
            pos = t.transform.translation
            rot = t.transform.rotation
            bMe = np.eye(4)
            bMe[:3, 3] = [pos.x, pos.y, pos.z]
            bMe[:3, :3] = Rot.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
        except Exception:
            # Fallback to our FK
            bMe = forward_kinematics(q, self.robot_model)

        bMc = bMe @ self.eMc
        oMc = np.linalg.inv(self.bMo) @ bMc
        cMo = np.linalg.inv(oMc)
        return cMo, oMc

    def _camera_jacobian(self, q):
        bMe = forward_kinematics(q, self.robot_model)
        bMc = bMe @ self.eMc
        cRb = bMc[:3, :3].T
        cWb = np.zeros((6, 6))
        cWb[:3, :3] = cRb
        cWb[3:, 3:] = cRb
        J_base = geometric_jacobian(q, self.robot_model)
        return cWb @ J_base

    def _se3_distance(self, c2w_a, c2w_b):
        t = np.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3])
        R = c2w_a[:3, :3].T @ c2w_b[:3, :3]
        r = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        return t, r

    def joint_cb(self, msg):
        q = np.zeros(6)
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                q[i] = msg.position[msg.name.index(name)]
        self.current_q = q

    def loop_cb(self):
        if self.current_q is None or not self.running:
            return

        # First time: compute bMo from actual initial EE pose via TF
        if self.bMo is None:
            self.q_init = self.current_q.copy()
            try:
                t = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
                pos = t.transform.translation
                rot = t.transform.rotation
                bMe = np.eye(4)
                bMe[:3, 3] = [pos.x, pos.y, pos.z]
                bMe[:3, :3] = Rot.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
            except Exception:
                bMe = forward_kinematics(self.q_init, self.robot_model)

            bMc = bMe @ self.eMc
            oMb = self.c2w_start @ np.linalg.inv(bMc)
            self.bMo = np.linalg.inv(oMb)
            self.get_logger().info(
                f'q_init = [{",".join(f"{np.degrees(q):.0f}" for q in self.q_init)}] deg')
            self.get_logger().info(f'EE pos: [{bMe[0,3]:.3f}, {bMe[1,3]:.3f}, {bMe[2,3]:.3f}]')

            # Initialize SimulatorCamera at start pose
            cMo_start = np.linalg.inv(self.c2w_start)
            self._cMo = cMo_start.copy()
            self._sim_camera = SimulatorCamera()
            self._sim_camera.setPosition(np.linalg.inv(cMo_start))
            self._sim_camera.setRobotState(1)

            self.get_logger().info('bMo computed. SimulatorCamera initialized. Starting VS...')

        self._do_vs()

    def _do_vs(self):
        """Main VS iteration."""
        q = self.current_q.copy()

        # Camera pose from SimulatorCamera (not from joints — avoids Jacobian drift)
        cMo = self._cMo
        c2w_cur = np.linalg.inv(cMo)

        # Render (PGM renders at original scales, inflated at scaled)
        render_scales = self.scales_original if self.is_pgm else self.scales_cur
        rgb_cur, gray_cur, depth_cur = self._render_with_scales(cMo, render_scales)

        # Features
        if self.is_pgm:
            cur_lam = self.pgm_lambdas[min(self.pgm_level, len(self.pgm_lambdas)-1)]
            s = self.FeaturePGM(lambda_g=cur_lam, border=10, device=self.device)
        else:
            s = create_feature('pinhole', device=self.device, border=10)
        s.init(self.H, self.W)
        s.setCameraParameters(self.cam_params)
        s.buildFrom(gray_cur, depth_cur)

        error = s.error(self.s_star)
        err_norm = torch.sum(error ** 2).item()

        c2w_goal = self.camtoworlds[self.goal_idx]
        t_err, r_err = self._se3_distance(c2w_cur, c2w_goal)

        # Log every 10 iterations
        if self.iteration % 10 == 0:
            self.get_logger().info(
                f'it={self.iteration} err={err_norm:.0f} t={t_err:.4f}m r={r_err:.2f}deg '
                f'q=[{",".join(f"{np.degrees(qi):.0f}" for qi in q)}]')

        # Publish errors
        err_msg = Float64()
        err_msg.data = err_norm
        self.pub_error.publish(err_msg)
        t_msg = Float64()
        t_msg.data = t_err
        self.pub_pose_t.publish(t_msg)
        r_msg = Float64()
        r_msg.data = r_err
        self.pub_pose_r.publish(r_msg)

        # Publish actual camera pose (c2w as flat 16-element array)
        pose_msg = Float64MultiArray()
        pose_msg.data = c2w_cur.flatten().tolist()
        self.pub_camera_pose.publish(pose_msg)

        # Publish images
        if self.is_pgm:
            # PGM mode: publish jet colormap of PGM features
            import cv2
            border = 10
            h_crop, w_crop = self.H - 2*border, self.W - 2*border
            pgm_cur_2d = s.G.reshape(h_crop, w_crop).cpu().numpy()
            pgm_des_2d = self.s_star.G.reshape(h_crop, w_crop).cpu().numpy()
            pgm_max = max(pgm_cur_2d.max(), pgm_des_2d.max(), 1e-8)

            # Current PGM → jet
            cur_jet = cv2.applyColorMap(
                (pgm_cur_2d / pgm_max * 255).clip(0, 255).astype(np.uint8),
                cv2.COLORMAP_JET)
            cur_jet_rgb = cv2.cvtColor(cur_jet, cv2.COLOR_BGR2RGB)
            self.pub_image.publish(self.bridge.cv2_to_imgmsg(cur_jet_rgb, encoding='rgb8'))

            # Desired PGM → jet
            des_jet = cv2.applyColorMap(
                (pgm_des_2d / pgm_max * 255).clip(0, 255).astype(np.uint8),
                cv2.COLORMAP_JET)
            des_jet_rgb = cv2.cvtColor(des_jet, cv2.COLOR_BGR2RGB)
            self.pub_desired.publish(self.bridge.cv2_to_imgmsg(des_jet_rgb, encoding='rgb8'))

            # Diff PGM → hot
            pgm_diff = np.abs(pgm_cur_2d - pgm_des_2d)
            diff_hot = cv2.applyColorMap(
                (pgm_diff / (pgm_max * 0.3 + 1e-8) * 255).clip(0, 255).astype(np.uint8),
                cv2.COLORMAP_HOT)
            diff_hot_rgb = cv2.cvtColor(diff_hot, cv2.COLOR_BGR2RGB)
            self.pub_diff.publish(self.bridge.cv2_to_imgmsg(diff_hot_rgb, encoding='rgb8'))
        else:
            # Standard mode: publish RGB renders
            img_cur = (rgb_cur.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            self.pub_image.publish(self.bridge.cv2_to_imgmsg(img_cur, encoding='rgb8'))
            img_des = (self.rgb_des_np * 255).clip(0, 255).astype(np.uint8)
            self.pub_desired.publish(self.bridge.cv2_to_imgmsg(img_des, encoding='rgb8'))
            diff = np.clip(np.abs(rgb_cur.cpu().numpy() - self.rgb_des_np) * 3, 0, 1)
            img_diff = (diff * 255).clip(0, 255).astype(np.uint8)
            self.pub_diff.publish(self.bridge.cv2_to_imgmsg(img_diff, encoding='rgb8'))

        # Max iterations
        if self.max_iter > 0 and self.iteration >= self.max_iter:
            self.get_logger().info(f'MAX ITER reached at it={self.iteration}')
            self._stop()
            return

        # Convergence
        if self.convergence_threshold > 0 and err_norm < self.convergence_threshold:
            self.get_logger().info(f'CONVERGED at it={self.iteration}!')
            self._stop()
            return

        # Camera velocity
        if self.is_pgm:
            # PGM: 6-DoF LM with PGM gain and stall detection
            LG = s.interaction()
            LtL = LG.T @ LG
            Hess = torch.linalg.inv(
                0.01 * torch.diag(torch.diag(LtL)) + LtL
                + 1e-6 * torch.eye(6, device=self.device))
            v_cam = -self.pgm_gain * (Hess @ LG.T @ error)

            # Stall detection → next lambda level
            mse = err_norm / max(error.shape[0], 1)
            if mse >= self.pgm_prev_mse * 0.999:
                self.pgm_stall += 1
            else:
                self.pgm_stall = 0
            self.pgm_prev_mse = mse
            if self.pgm_stall > 50 and self.pgm_level < len(self.pgm_lambdas) - 1:
                self.pgm_level += 1
                self.pgm_stall = 0
                new_lam = self.pgm_lambdas[self.pgm_level]
                self.s_star = self.FeaturePGM(lambda_g=new_lam, border=10, device=self.device)
                self.s_star.init(self.H, self.W)
                self.s_star.setCameraParameters(self.cam_params)
                self.s_star.buildFrom(self.gray_des, self.depth_des)
                self.get_logger().info(f'PGM level {self.pgm_level}: lambda_g={new_lam}')
        else:
            # Original / Inflated: standard LM
            Ls = s.interaction()
            Hs = Ls.T @ Ls
            diagHs = torch.diag(torch.diag(Hs))
            Hess = torch.linalg.inv(
                self.mu * diagHs + Hs + 1e-6 * torch.eye(6, device=self.device))
            v_cam = -self.gain * (Hess @ Ls.T @ error)
        v_np = v_cam.detach().cpu().numpy()

        # Clamp camera velocity
        vt = np.linalg.norm(v_np[:3])
        vr = np.linalg.norm(v_np[3:])
        if vt > 0.5: v_np[:3] *= 0.5 / vt
        if vr > 0.3: v_np[3:] *= 0.3 / vr

        # Update camera pose via SimulatorCamera (proven to work)
        self._sim_camera.setVelocity("camera", v_np)
        wMc = self._sim_camera.getPosition()
        self._cMo = np.linalg.inv(wMc)

        # Compute corresponding joint velocity for robot visualization
        J_cam = self._camera_jacobian(q)
        damping = 0.01
        JJt = J_cam @ J_cam.T + damping**2 * np.eye(6)
        dq = J_cam.T @ np.linalg.solve(JJt, v_np)

        # Clamp joint velocities
        max_jvel = 1.0
        scale = np.max(np.abs(dq) / max_jvel)
        if scale > 1.0:
            dq /= scale

        # Publish joint velocities (for RViz visualization)
        msg = Float64MultiArray()
        msg.data = dq.tolist()
        self.pub_vel.publish(msg)

        self.iteration += 1

    def _stop(self):
        self.running = False
        msg = Float64MultiArray()
        msg.data = [0.0] * 6
        self.pub_vel.publish(msg)
        # Shutdown the node so the process exits
        raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = VSNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
