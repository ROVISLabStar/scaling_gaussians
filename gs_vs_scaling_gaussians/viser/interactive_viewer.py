"""
Interactive 3D Viewer for Visual Servoing with 3DGS
=====================================================

Viser-based viewer with:
- Full 3DGS Gaussian splat rendering
- Camera frustums with rendered images (desired, start, current)
- Diff image toggled on the current frustum (same plane)
- Multi-run comparison: overlaid trajectories and error/pose plots
- VS modes: original, inflated, coarse-to-fine, PGM-VS

Usage:
    python -m gs_vs_scaling_gaussians.viser.interactive_viewer \
        --ckpt <checkpoint> --cfg <config> --port 8080

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
import plotly.graph_objects as go

from gs_vs.simulators.RobotSimulator import SimulatorCamera
from gs_vs.simulators.CameraParameters import CameraParameters
from gs_vs.features.factory import create_feature
from gs_vs.datasets.colmap import Parser
from gsplat.rendering import rasterization


def se3_distance(c2w_a, c2w_b):
    t_dist = np.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3])
    R_rel = c2w_a[:3, :3].T @ c2w_b[:3, :3]
    angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
    return t_dist, np.degrees(angle)


@torch.no_grad()
def render_gsplat(cMo, means, quats, scales, opacities, colors,
                  sh_degree, K_np, W, H, device="cuda"):
    viewmat = torch.from_numpy(cMo).float().to(device)[None]
    Ks = torch.from_numpy(K_np).float().to(device)[None]
    renders, _, _ = rasterization(
        means=means, quats=quats, scales=scales,
        opacities=opacities, colors=colors,
        sh_degree=sh_degree, viewmats=viewmat, Ks=Ks,
        width=W, height=H, packed=True,
        render_mode="RGB+ED", camera_model="pinhole",
    )
    rgb = torch.clamp(renders[0, ..., :3], 0.0, 1.0)
    depth = renders[0, ..., 3]
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return rgb, gray, depth


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


def main():
    p = argparse.ArgumentParser(
        description="Interactive VS Viewer with 3DGS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--data_factor", type=int, default=4)
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    device = "cuda"

    # ---- Load scene ----
    cfg = load_basic_cfg_fields(args.cfg)
    parser = Parser(data_dir=cfg["data_dir"], factor=cfg["data_factor"],
                    normalize=cfg["normalize_world_space"], test_every=8)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    splats = ckpt["splats"]
    means_t = splats["means"].to(device)
    quats_t = splats["quats"].to(device)
    scales_original = torch.exp(splats["scales"]).to(device)
    opacities_t = torch.sigmoid(splats["opacities"]).to(device)
    colors_t = torch.cat([splats["sh0"], splats["shN"]], dim=1).to(device)
    sh_degree = int(np.sqrt(colors_t.shape[1]) - 1)

    camtoworlds = parser.camtoworlds
    n_views = len(camtoworlds)
    W_full, H_full = list(parser.imsize_dict.values())[0]
    K_colmap = list(parser.Ks_dict.values())[0]

    df = args.data_factor
    W, H = W_full // df, H_full // df
    fx, fy = K_colmap[0, 0] / df, K_colmap[1, 1] / df
    cx, cy = K_colmap[0, 2] / df, K_colmap[1, 2] / df
    K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    cam_params = CameraParameters(px=fx, py=fy, u0=cx, v0=cy)

    pixel_ratio = (W * H) / (W_full * H_full)
    convergence_threshold = 10000 * pixel_ratio

    print(f"[Scene] {n_views} views, {W}x{H}")

    # ---- Splat data for viser ----
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

    # ---- Viser server ----
    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_layout="collapsible", control_width="large")
    print(f"[Viser] http://localhost:{args.port}")

    # Set up direction
    up_world = np.linalg.inv(camtoworlds[0])[:3, :3] @ np.array([0.0, -1.0, 0.0])
    up_world /= np.linalg.norm(up_world)
    server.scene.set_up_direction(tuple(up_world))

    # ---- 3DGS Gaussian Splats ----
    splats_handle = [None]

    def show_splats():
        if splats_handle[0] is None:
            print(f"[Splats] Loading {means_np.shape[0]} Gaussians...")
            splats_handle[0] = server.scene.add_gaussian_splats(
                "/splats",
                centers=means_np,
                rgbs=rgbs_np,
                opacities=opacities_np[:, None],
                covariances=covariances,
            )
            print(f"[Splats] Loaded.")

    def hide_splats():
        if splats_handle[0] is not None:
            splats_handle[0].remove()
            splats_handle[0] = None

    show_splats()

    # ---- Training camera frustums with IDs ----
    for i in range(0, n_views, max(1, n_views // 20)):
        wxyz = tf.SO3.from_matrix(camtoworlds[i][:3, :3]).wxyz
        pos = camtoworlds[i][:3, 3]
        server.scene.add_camera_frustum(
            f"/scene/cameras/view_{i}",
            fov=np.pi / 3, aspect=W / H, scale=0.02,
            wxyz=wxyz, position=pos,
            color=(128, 128, 128))
        server.scene.add_label(
            f"/scene/cameras/label_{i}",
            text=str(i),
            position=pos + np.array([0, 0, 0.02]),
        )

    # ---- Frustums with images ----
    fov_rad = 2 * np.arctan(W / (2 * fx))
    downsample = 2
    image_scale = 0.15

    def render_view(view_idx, sf=1.0):
        c2w = camtoworlds[view_idx]
        cMo = np.linalg.inv(c2w)
        rgb, _, _ = render_gsplat(cMo, means_t, quats_t, scales_original * sf,
                                   opacities_t, colors_t, sh_degree, K_np, W, H, device)
        return rgb.cpu().numpy()

    si_init, gi_init = 14, 10
    img_start = render_view(si_init)
    img_goal = render_view(gi_init)

    # Desired frustum (green)
    frustum_goal = server.scene.add_camera_frustum(
        "/frustums/goal", fov=fov_rad, aspect=W/H, scale=image_scale,
        image=img_goal[::downsample, ::downsample],
        wxyz=tf.SO3.from_matrix(camtoworlds[gi_init][:3, :3]).wxyz,
        position=camtoworlds[gi_init][:3, 3], color=(0, 200, 0))
    server.scene.add_frame("/frustums/goal/axes", axes_length=0.08, axes_radius=0.003)

    # Start frustum (red)
    frustum_start = server.scene.add_camera_frustum(
        "/frustums/start", fov=fov_rad, aspect=W/H, scale=image_scale,
        image=img_start[::downsample, ::downsample],
        wxyz=tf.SO3.from_matrix(camtoworlds[si_init][:3, :3]).wxyz,
        position=camtoworlds[si_init][:3, 3], color=(200, 80, 80))
    server.scene.add_frame("/frustums/start/axes", axes_length=0.08, axes_radius=0.003)

    # Current frustum (blue) — shows current image or diff
    frustum_current = server.scene.add_camera_frustum(
        "/frustums/current", fov=fov_rad, aspect=W/H, scale=image_scale,
        image=img_start[::downsample, ::downsample],
        wxyz=tf.SO3.from_matrix(camtoworlds[si_init][:3, :3]).wxyz,
        position=camtoworlds[si_init][:3, 3], color=(80, 80, 200))
    server.scene.add_frame("/frustums/current/axes", axes_length=0.08, axes_radius=0.003)

    # Store images for diff toggle
    current_rgb = [img_start.copy()]
    desired_rgb = [img_goal.copy()]

    # ---- Trajectory colors ----
    traj_colors = {
        "original": (200, 200, 200), "inflated": (0, 150, 255),
        "coarse_to_fine": (0, 200, 100), "pgm_vs": (255, 150, 0),
    }
    plot_colors = {
        "original": "gray", "inflated": "blue",
        "coarse_to_fine": "green", "pgm_vs": "orange",
    }
    stored_trajectories = {}
    run_history = {}
    traj_handles = {}  # mode -> scene handle

    # ---- Plotly plots ----
    fig_error = go.Figure()
    fig_error.update_layout(title="Photometric Error", xaxis_title="Iteration",
                            yaxis_title="||e||^2", yaxis_type="log",
                            margin=dict(l=20, r=20, t=40, b=20))
    error_plot = server.gui.add_plotly(fig_error, aspect=1.4)

    fig_pose = go.Figure()
    fig_pose.update_layout(title="Pose Error", xaxis_title="Iteration",
                           yaxis_title="Error",
                           margin=dict(l=20, r=20, t=40, b=20))
    pose_plot = server.gui.add_plotly(fig_pose, aspect=1.4)

    fig_vel = go.Figure()
    for lbl in ["vx", "vy", "vz", "wx", "wy", "wz"]:
        fig_vel.add_trace(go.Scatter(x=[], y=[], mode="lines", name=lbl))
    fig_vel.update_layout(title="Velocity (current)", xaxis_title="Iteration",
                          yaxis_title="Vel", margin=dict(l=20, r=20, t=40, b=20))
    vel_plot = server.gui.add_plotly(fig_vel, aspect=1.4)

    # ---- GUI ----
    with server.gui.add_folder("Visual Servoing"):
        start_slider = server.gui.add_slider("Start View", 0, n_views-1, 1, si_init)
        goal_slider = server.gui.add_slider("Goal View", 0, n_views-1, 1, gi_init)
        mode_dropdown = server.gui.add_dropdown(
            "Mode", options=["original", "inflated", "coarse_to_fine", "pgm_vs"])
        max_iter_slider = server.gui.add_slider("Max Iter", 100, 5000, 100, 2000)
        pass  # convergence thresholds are in mode-specific folders
        run_button = server.gui.add_button("Run VS")
        stop_button = server.gui.add_button("Stop")
        clear_button = server.gui.add_button("Clear Trajectories")
        status_text = server.gui.add_text("Status", initial_value="Ready")
        iter_text = server.gui.add_text("Iter", initial_value="--")
        error_text = server.gui.add_text("Error", initial_value="--")
        pose_text = server.gui.add_text("Pose", initial_value="--")

    # ---- Mode-specific parameter folders ----
    # Original PVS parameters
    folder_original = server.gui.add_folder("Original PVS Parameters")
    with folder_original:
        orig_gain = server.gui.add_slider("Gain (lambda)", 1.0, 30.0, 1.0, 10.0)
        orig_mu = server.gui.add_slider("LM Damping (mu)", 0.001, 0.1, 0.001, 0.01)
        orig_thresh = server.gui.add_slider("Conv. Threshold", 0, 50000, 10, 100)

    # Inflated PVS parameters
    folder_inflated = server.gui.add_folder("Inflated PVS Parameters")
    with folder_inflated:
        inf_scale = server.gui.add_slider("Scale Factor (alpha)", 1.0, 5.0, 0.1, 1.8)
        inf_gain = server.gui.add_slider("Gain (lambda)", 1.0, 30.0, 1.0, 10.0)
        inf_mu = server.gui.add_slider("LM Damping (mu)", 0.001, 0.1, 0.001, 0.01)
        inf_thresh = server.gui.add_slider("Conv. Threshold", 0, 50000, 10, 0)

    # Coarse-to-fine parameters
    folder_c2f = server.gui.add_folder("Coarse-to-Fine Parameters")
    with folder_c2f:
        c2f_scale = server.gui.add_slider("Initial Scale Factor", 1.0, 5.0, 0.1, 1.8)
        c2f_phases = server.gui.add_slider("Num Phases", 2, 10, 1, 5)
        c2f_gain = server.gui.add_slider("Gain (lambda)", 1.0, 30.0, 1.0, 10.0)
        c2f_mu = server.gui.add_slider("LM Damping (mu)", 0.001, 0.1, 0.001, 0.01)
        c2f_thresh = server.gui.add_slider("Conv. Threshold", 0, 50000, 10, 100)

    # PGM-VS parameters
    folder_pgm = server.gui.add_folder("PGM-VS Parameters")
    with folder_pgm:
        pgm_lambda_init = server.gui.add_slider("Lambda_g Initial", 1.0, 30.0, 0.5, 5.0)
        pgm_lambda_final = server.gui.add_slider("Lambda_g Final", 0.5, 5.0, 0.5, 1.0)
        pgm_gain = server.gui.add_slider("Gain", 1.0, 30.0, 1.0, 10.0)
        pgm_stall_thresh = server.gui.add_slider("Stall Threshold (iters)", 10, 100, 10, 50)
        pgm_thresh = server.gui.add_slider("Conv. Threshold", 0, 50000, 10, 100)

    # Show/hide folders based on mode
    def update_mode_folders():
        mode = mode_dropdown.value
        folder_original.visible = (mode == "original")
        folder_inflated.visible = (mode == "inflated")
        folder_c2f.visible = (mode == "coarse_to_fine")
        folder_pgm.visible = (mode == "pgm_vs")

    update_mode_folders()

    @mode_dropdown.on_update
    def _(_): update_mode_folders()

    with server.gui.add_folder("Display"):
        show_splats_cb = server.gui.add_checkbox("Show Splats", True)
        show_diff_cb = server.gui.add_checkbox("Show Diff Image", False)
        show_goal_cb = server.gui.add_checkbox("Show Goal Frustum", True)
        show_start_cb = server.gui.add_checkbox("Show Start Frustum", True)
        show_current_cb = server.gui.add_checkbox("Show Current Frustum", True)
        show_training_cb = server.gui.add_checkbox("Show Training Cameras", True)

    # ---- State ----
    vs_running = [False]

    def update_frustum_image():
        """Toggle current frustum between rendered image and diff."""
        ds = downsample
        if show_diff_cb.value:
            diff = np.clip(np.abs(current_rgb[0] - desired_rgb[0]) * 3, 0, 1)
            frustum_current.image = diff[::ds, ::ds]
        else:
            frustum_current.image = current_rgb[0][::ds, ::ds]

    def update_display():
        si, gi = start_slider.value, goal_slider.value
        img_s = render_view(si)
        img_g = render_view(gi)
        current_rgb[0] = img_s
        desired_rgb[0] = img_g
        frustum_start.image = img_s[::downsample, ::downsample]
        frustum_start.wxyz = tf.SO3.from_matrix(camtoworlds[si][:3, :3]).wxyz
        frustum_start.position = camtoworlds[si][:3, 3]
        frustum_goal.image = img_g[::downsample, ::downsample]
        frustum_goal.wxyz = tf.SO3.from_matrix(camtoworlds[gi][:3, :3]).wxyz
        frustum_goal.position = camtoworlds[gi][:3, 3]
        frustum_current.wxyz = tf.SO3.from_matrix(camtoworlds[si][:3, :3]).wxyz
        frustum_current.position = camtoworlds[si][:3, 3]
        update_frustum_image()
        t, r = se3_distance(camtoworlds[si], camtoworlds[gi])
        status_text.value = f"Ready | d={t:.3f}m r={r:.1f}deg"

    update_display()

    @goal_slider.on_update
    def _(_): update_display()
    @start_slider.on_update
    def _(_): update_display()
    @show_diff_cb.on_update
    def _(_): update_frustum_image()
    @show_splats_cb.on_update
    def _(_):
        if show_splats_cb.value:
            show_splats()
        else:
            hide_splats()

    @show_goal_cb.on_update
    def _(_): frustum_goal.visible = show_goal_cb.value

    @show_start_cb.on_update
    def _(_): frustum_start.visible = show_start_cb.value

    @show_current_cb.on_update
    def _(_): frustum_current.visible = show_current_cb.value
    @stop_button.on_click
    def _(_): vs_running[0] = False; status_text.value = "Stopped"
    @clear_button.on_click
    def _(_):
        stored_trajectories.clear(); run_history.clear()
        for m, handle in list(traj_handles.items()):
            handle.remove()
        traj_handles.clear()
        fig_error.data = []; error_plot.figure = fig_error
        fig_pose.data = []; pose_plot.figure = fig_pose
        status_text.value = "Cleared"

    @run_button.on_click
    def _(_):
        if vs_running[0]: return
        vs_running[0] = True
        threading.Thread(target=run_vs, daemon=True).start()

    def run_vs():
        import sys
        sys.path.insert(0, os.path.join(
            os.path.dirname(__file__), '..', '..', 'gs_vs_pgm_vs'))

        si, gi = start_slider.value, goal_slider.value
        mode = mode_dropdown.value
        max_iter = int(max_iter_slider.value)
        is_pgm = (mode == "pgm_vs")

        # Read mode-specific parameters
        if mode == "original":
            gain, mu = orig_gain.value, orig_mu.value
            scale_factor, sf = 1.0, 1.0
            conv_threshold = orig_thresh.value
        elif mode == "inflated":
            gain, mu = inf_gain.value, inf_mu.value
            scale_factor = inf_scale.value
            sf = scale_factor
            conv_threshold = inf_thresh.value
        elif mode == "coarse_to_fine":
            gain, mu = c2f_gain.value, c2f_mu.value
            scale_factor = c2f_scale.value
            sf = scale_factor
            conv_threshold = c2f_thresh.value
        elif mode == "pgm_vs":
            gain, mu = pgm_gain.value, 0.01
            scale_factor, sf = 1.0, 1.0
            conv_threshold = pgm_thresh.value

        c2w_goal, c2w_start = camtoworlds[gi], camtoworlds[si]
        cMo_start, cMo_goal = np.linalg.inv(c2w_start), np.linalg.inv(c2w_goal)
        status_text.value = f"Running {mode}..."

        # Desired features (at current scale — same as what VS minimizes)
        scales_cur = scales_original * sf
        rgb_des, gray_des, depth_des = render_gsplat(
            cMo_goal, means_t, quats_t, scales_cur, opacities_t,
            colors_t, sh_degree, K_np, W, H, device)
        desired_rgb[0] = rgb_des.cpu().numpy()

        if is_pgm:
            from features.FeaturePGM import FeaturePGM
            # Build lambda levels from initial to final
            lam_init = pgm_lambda_init.value
            lam_final = pgm_lambda_final.value
            pgm_lambdas = []
            lam = lam_init
            while lam > 2.0 * lam_final:
                pgm_lambdas.append(lam)
                lam *= 0.5
            pgm_lambdas.append(lam_final)
            pgm_level, pgm_stall_count, pgm_prev = [0], [0], [float('inf')]
            pgm_stall_limit = int(pgm_stall_thresh.value)
            s_star = FeaturePGM(lambda_g=pgm_lambdas[0], border=10, device=device)
        else:
            s_star = create_feature("pinhole", device=device, border=10)
        s_star.init(H, W); s_star.setCameraParameters(cam_params)
        s_star.buildFrom(gray_des, depth_des)

        wMo = np.eye(4)
        robot = SimulatorCamera()
        robot.setPosition(wMo @ np.linalg.inv(cMo_start))
        robot.setRobotState(1)
        cMo = cMo_start.copy()

        if mode == "coarse_to_fine":
            n_ph = int(c2f_phases.value)
            sched = np.ones(max_iter)
            ipp = max_iter // n_ph
            for ph in range(n_ph):
                s0 = ph * ipp
                s1 = (ph+1)*ipp if ph < n_ph-1 else max_iter
                sched[s0:s1] = scale_factor ** (1.0 - ph/(n_ph-1))

        iters, errs, t_hist, r_hist, vels, trajectory = [], [], [], [], [], []
        tc = traj_colors.get(mode, (128, 128, 128))

        for it in range(max_iter):
            if not vs_running[0]: break

            if mode == "coarse_to_fine":
                sf = sched[it]; scales_cur = scales_original * sf
                _, gd, dd = render_gsplat(cMo_goal, means_t, quats_t, scales_cur,
                    opacities_t, colors_t, sh_degree, K_np, W, H, device)
                s_star = create_feature("pinhole", device=device, border=10)
                s_star.init(H, W); s_star.setCameraParameters(cam_params)
                s_star.buildFrom(gd, dd)

            render_scales = scales_original if is_pgm else scales_cur
            rgb_cur, gray_cur, depth_cur = render_gsplat(
                cMo, means_t, quats_t, render_scales, opacities_t,
                colors_t, sh_degree, K_np, W, H, device)

            if is_pgm:
                from features.FeaturePGM import FeaturePGM
                lam = pgm_lambdas[min(pgm_level[0], len(pgm_lambdas)-1)]
                s = FeaturePGM(lambda_g=lam, border=10, device=device)
            else:
                s = create_feature("pinhole", device=device, border=10)
            s.init(H, W); s.setCameraParameters(cam_params)
            s.buildFrom(gray_cur, depth_cur)

            error = s.error(s_star)
            err_norm = torch.sum(error**2).item()
            c2w_cur = np.linalg.inv(cMo)
            t_err, r_err = se3_distance(c2w_cur, c2w_goal)

            iters.append(it); errs.append(err_norm)
            t_hist.append(t_err); r_hist.append(r_err)
            trajectory.append(c2w_cur[:3, 3].copy())

            iter_text.value = f"{it}/{max_iter}"
            error_text.value = f"{err_norm:.0f}"
            pose_text.value = f"t={t_err:.4f}m r={r_err:.2f}deg"

            # Update current frustum (image or diff on same plane)
            rgb_cur_np = rgb_cur.cpu().numpy()
            current_rgb[0] = rgb_cur_np
            update_frustum_image()
            frustum_current.wxyz = tf.SO3.from_matrix(c2w_cur[:3, :3]).wxyz
            frustum_current.position = c2w_cur[:3, 3]

            # Trajectory
            if len(trajectory) > 1:
                pts = np.array(trajectory)
                traj_handles[mode] = server.scene.add_point_cloud(
                    f"/scene/traj_{mode}", points=pts,
                    colors=np.full((len(pts), 3), tc, dtype=np.uint8),
                    point_size=0.005)

            # Convergence (photometric error only)
            if conv_threshold > 0 and err_norm < conv_threshold * pixel_ratio:
                status_text.value = f"CONVERGED it={it} ({mode})"; vs_running[0] = False; break

            # Velocity
            if is_pgm:
                LG = s.interaction(); LtL = LG.T @ LG
                H_inv = torch.linalg.inv(0.01*torch.diag(torch.diag(LtL)) + LtL
                    + 1e-6*torch.eye(6, device=device))
                v = -10.0 * (H_inv @ LG.T @ error)
                mse = err_norm / max(error.shape[0], 1)
                if mse >= pgm_prev[0] * 0.999: pgm_stall_count[0] += 1
                else: pgm_stall_count[0] = 0
                pgm_prev[0] = mse
                if pgm_stall_count[0] > pgm_stall_limit and pgm_level[0] < len(pgm_lambdas)-1:
                    pgm_level[0] += 1; pgm_stall_count[0] = 0
                    new_lam = pgm_lambdas[pgm_level[0]]
                    s_star = FeaturePGM(lambda_g=new_lam, border=10, device=device)
                    s_star.init(H, W); s_star.setCameraParameters(cam_params)
                    s_star.buildFrom(gray_des, depth_des)
            else:
                Ls = s.interaction(); Hs_mat = Ls.T @ Ls
                H_inv = torch.linalg.inv(mu*torch.diag(torch.diag(Hs_mat)) + Hs_mat
                    + 1e-6*torch.eye(6, device=device))
                v = -gain * (H_inv @ Ls.T @ error)

            v_np = v.detach().cpu().numpy()
            vt, vr = np.linalg.norm(v_np[:3]), np.linalg.norm(v_np[3:])
            if vt > 0.5: v_np[:3] *= 0.5 / vt
            if vr > 0.3: v_np[3:] *= 0.3 / vr
            vels.append(v_np.copy())

            robot.setVelocity("camera", v_np)
            wMc = robot.getPosition()
            cMo = np.linalg.inv(wMc) @ wMo

            # Live plots every 5 iters
            if it % 5 == 0:
                fig_error.data = []
                for m, h in run_history.items():
                    fig_error.add_trace(go.Scatter(x=h["i"], y=h["e"], mode="lines",
                        name=m, line=dict(color=plot_colors.get(m,"gray"), width=1, dash="dot")))
                fig_error.add_trace(go.Scatter(x=iters, y=errs, mode="lines",
                    name=f"{mode} (live)", line=dict(color=plot_colors.get(mode,"red"), width=3)))
                error_plot.figure = fig_error
                for j in range(6):
                    fig_vel.data[j].x = iters[:len(vels)]
                    fig_vel.data[j].y = [vv[j] for vv in vels]
                vel_plot.figure = fig_vel

            time.sleep(0.02)

        # Store
        stored_trajectories[mode] = trajectory
        run_history[mode] = {"i": iters, "e": errs, "t": t_hist, "r": r_hist}

        if vs_running[0]:
            status_text.value = f"FAILED {max_iter} it ({mode})"; vs_running[0] = False

        # Final comparison plots
        fig_error.data = []
        fig_pose.data = []
        for m, h in run_history.items():
            c = plot_colors.get(m, "gray")
            fig_error.add_trace(go.Scatter(x=h["i"], y=h["e"], mode="lines",
                name=m, line=dict(color=c, width=2)))
            fig_pose.add_trace(go.Scatter(x=h["i"], y=h["t"], mode="lines",
                name=f"{m} (t)", line=dict(color=c, width=2)))
            fig_pose.add_trace(go.Scatter(x=h["i"], y=h["r"], mode="lines",
                name=f"{m} (r)", line=dict(color=c, width=1, dash="dash")))
        error_plot.figure = fig_error
        pose_plot.figure = fig_pose
        for j in range(6):
            fig_vel.data[j].x = iters[:len(vels)]
            fig_vel.data[j].y = [vv[j] for vv in vels]
        vel_plot.figure = fig_vel

    print("[Viewer] Ready. Open browser to start.")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
