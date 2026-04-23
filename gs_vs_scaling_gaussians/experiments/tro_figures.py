"""
Generate TRO-quality figures for Scale-Adaptive PVS paper.
============================================================

Runs VS experiments and saves all data needed for publication figures
in the style of Crombez et al. TRO 2019:
  - Desired/initial/final RGB images + inflated renders
  - Difference images at key iterations
  - Photometric error vs iterations (all methods overlaid)
  - Scale parameter α vs iterations
  - Camera velocities (vx,vy,vz,ωx,ωy,ωz) vs iterations
  - 3D camera trajectory
  - Per-iteration camera poses

Usage:
    python -m gs_vs_scaling_gaussians.experiments.tro_figures \
        --ckpt <ckpt> --cfg <cfg> \
        --start_idx 94 --goal_idx 60 \
        --out_dir gs_vs_scaling_gaussians/logs/tro_figures

Author: Youssef ALJ (UM6P)
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gs_vs.simulators.RobotSimulator import SimulatorCamera
from gs_vs.simulators.CameraParameters import CameraParameters
from gs_vs.features.factory import create_feature
from gs_vs.datasets.colmap import Parser
from gsplat.rendering import rasterization


def load_cfg(path):
    data = {}
    with open(path) as f:
        for line in f:
            if ":" not in line: continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if k == "data_dir": data["data_dir"] = v
            elif k == "data_factor": data["data_factor"] = int(v)
            elif k == "normalize_world_space":
                data["normalize_world_space"] = v.lower() == "true"
    data.setdefault("data_factor", 1)
    data.setdefault("normalize_world_space", True)
    return data


@torch.no_grad()
def render(cMo, means, quats, scales, opacities, colors,
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


def run_servo(mode, sf, c2w_start, c2w_goal,
              means, quats, scales_orig, opacities, colors,
              sh_degree, K_np, W, H, cam_params,
              mu, gain, max_iter, device="cuda"):
    """Run VS and record full history."""

    scales = scales_orig * sf if mode != "original" else scales_orig

    cMo_goal = np.linalg.inv(c2w_goal)
    cMo_start = np.linalg.inv(c2w_start)

    # Desired
    rgb_des, gray_des, depth_des = render(
        cMo_goal, means, quats, scales, opacities, colors,
        sh_degree, K_np, W, H, device)
    s_star = create_feature("pinhole", device=device, border=10)
    s_star.init(H, W)
    s_star.setCameraParameters(cam_params)
    s_star.buildFrom(gray_des, depth_des)

    # Simulator
    wMo = np.eye(4)
    robot = SimulatorCamera()
    robot.setPosition(wMo @ np.linalg.inv(cMo_start))
    robot.setRobotState(1)
    cMo = cMo_start.copy()

    # History
    errors = []
    velocities = []
    poses = []         # c2w per iteration
    t_errors = []
    r_errors = []
    scale_history = []

    # For coarse-to-fine
    current_sf = sf
    switched = False

    for it in range(max_iter):
        # Coarse-to-fine: switch to sf=1.0 when error plateaus
        if mode == "coarse_to_fine" and not switched:
            if it > 50 and len(errors) > 10:
                recent = errors[-10:]
                if max(recent) - min(recent) < 0.5 * errors[0] * 0.01:
                    current_sf = 1.0
                    scales = scales_orig
                    # Re-render desired with original scales
                    rgb_des, gray_des, depth_des = render(
                        cMo_goal, means, quats, scales, opacities, colors,
                        sh_degree, K_np, W, H, device)
                    s_star = create_feature("pinhole", device=device, border=10)
                    s_star.init(H, W)
                    s_star.setCameraParameters(cam_params)
                    s_star.buildFrom(gray_des, depth_des)
                    switched = True

        scale_history.append(current_sf)

        rgb_cur, gray_cur, depth_cur = render(
            cMo, means, quats, scales, opacities, colors,
            sh_degree, K_np, W, H, device)

        s = create_feature("pinhole", device=device, border=10)
        s.init(H, W)
        s.setCameraParameters(cam_params)
        s.buildFrom(gray_cur, depth_cur)

        error = s.error(s_star)
        err_norm = torch.sum(error ** 2).item()
        errors.append(err_norm)

        c2w_cur = np.linalg.inv(cMo)
        poses.append(c2w_cur.copy())

        t_err = np.linalg.norm(c2w_cur[:3, 3] - c2w_goal[:3, 3])
        R_rel = c2w_cur[:3, :3].T @ c2w_goal[:3, :3]
        r_err = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
        t_errors.append(t_err)
        r_errors.append(r_err)

        # No early stopping: run full max_iter to show velocity stabilization
        # Record zero velocity if already converged
        if t_err < 0.005 and r_err < 0.5:
            velocities.append(np.zeros(6))
            continue

        # LM update
        Ls = s.interaction()
        Hs = Ls.T @ Ls
        diagHs = torch.diag(torch.diag(Hs))
        Hess = torch.linalg.inv(mu * diagHs + Hs + 1e-6 * torch.eye(6, device=device))
        v = -gain * (Hess @ Ls.T @ error)
        v_np = v.detach().cpu().numpy()

        # Velocity clamping
        vt = np.linalg.norm(v_np[:3])
        vr = np.linalg.norm(v_np[3:])
        if vt > 0.5: v_np[:3] *= 0.5 / vt
        if vr > 0.3: v_np[3:] *= 0.3 / vr

        velocities.append(v_np.copy())

        robot.setVelocity("camera", v_np)
        wMc = robot.getPosition()
        cMo = np.linalg.inv(wMc) @ wMo

    n_iter = len(errors)
    print(f"  [{mode} sf={sf}] {n_iter} iterations, "
          f"final: err={errors[-1]:.0f}, t={t_errors[-1]:.4f}m, r={r_errors[-1]:.1f}°")

    return {
        "mode": mode,
        "sf": sf,
        "errors": np.array(errors),
        "velocities": np.array(velocities) if velocities else np.zeros((0, 6)),
        "poses": np.array(poses),
        "t_errors": np.array(t_errors),
        "r_errors": np.array(r_errors),
        "scale_history": np.array(scale_history),
        "n_iter": n_iter,
        "converged": t_errors[-1] < 0.01 and r_errors[-1] < 1.0,
    }


def save_images(cMo_start, cMo_goal, means, quats, scales_orig, opacities,
                colors, sh_degree, K_np, W, H, sf, out_dir, device="cuda"):
    """Save desired, initial, final images + inflated versions."""
    os.makedirs(out_dir, exist_ok=True)
    scales_inf = scales_orig * sf

    for label, cMo, sc in [("desired", cMo_goal, scales_orig),
                            ("desired_inflated", cMo_goal, scales_inf),
                            ("initial", cMo_start, scales_orig),
                            ("initial_inflated", cMo_start, scales_inf)]:
        rgb, _, _ = render(cMo, means, quats, sc, opacities, colors,
                          sh_degree, K_np, W, H, device)
        img = (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{label}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def plot_experiment_panel(results_dict, out_dir, pair_name):
    """Generate Crombez-style experiment panel figure."""
    os.makedirs(out_dir, exist_ok=True)

    methods = list(results_dict.keys())
    colors_map = {
        "original": "#d62728",
        "inflated": "#1f77b4",
        "coarse_to_fine": "#2ca02c",
        "inflated_1.2": "#ff7f0e",
        "inflated_1.4": "#9467bd",
        "inflated_1.8": "#1f77b4",
        "inflated_2.0": "#8c564b",
    }

    # ─── Figure 1: Error curves (all methods overlaid) ───
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    for name, res in results_dict.items():
        color = colors_map.get(name, None)
        label = name.replace("_", " ").title()
        if name.startswith("inflated"):
            label = f"Inflated (α={res['sf']})"
        ax.plot(res["errors"], label=label, color=color, linewidth=1.5)
    ax.set_xlabel("Iterations", fontsize=11)
    ax.set_ylabel("Residual error $\\|\\mathbf{e}\\|^2$", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Pair {pair_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_curves.pdf"), dpi=300)
    fig.savefig(os.path.join(out_dir, "error_curves.png"), dpi=200)
    plt.close(fig)

    # ─── Figure 2: Pose error (translation + rotation) ───
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    for name, res in results_dict.items():
        color = colors_map.get(name, None)
        label = name.replace("_", " ").title()
        if name.startswith("inflated"):
            label = f"Inflated (α={res['sf']})"
        ax1.plot(res["t_errors"], label=label, color=color, linewidth=1.5)
        ax2.plot(res["r_errors"], label=label, color=color, linewidth=1.5)
    ax1.set_xlabel("Iterations"); ax1.set_ylabel("Translation error (m)")
    ax2.set_xlabel("Iterations"); ax2.set_ylabel("Rotation error (°)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pose_errors.pdf"), dpi=300)
    fig.savefig(os.path.join(out_dir, "pose_errors.png"), dpi=200)
    plt.close(fig)

    # ─── Figure 3: Scale parameter evolution ───
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    for name, res in results_dict.items():
        if np.all(res["scale_history"] == 1.0):
            continue  # skip original
        color = colors_map.get(name, None)
        label = name.replace("_", " ").title()
        if name.startswith("inflated"):
            label = f"α={res['sf']}"
        ax.plot(res["scale_history"], label=label, color=color, linewidth=1.5)
    ax.set_xlabel("Iterations", fontsize=11)
    ax.set_ylabel("Scale parameter $\\alpha$", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scale_evolution.pdf"), dpi=300)
    fig.savefig(os.path.join(out_dir, "scale_evolution.png"), dpi=200)
    plt.close(fig)

    # ─── Figure 4: Velocities (for each method) ───
    for name, res in results_dict.items():
        if len(res["velocities"]) == 0:
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
        v = res["velocities"]
        iters = np.arange(len(v))

        ax1.plot(iters, v[:, 0], label="$v_x$", linewidth=1.2)
        ax1.plot(iters, v[:, 1], label="$v_y$", linewidth=1.2)
        ax1.plot(iters, v[:, 2], label="$v_z$", linewidth=1.2)
        ax1.set_xlabel("Iterations"); ax1.set_ylabel("Translational velocities (m/s)")
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

        ax2.plot(iters, v[:, 3], label="$\\omega_x$", linewidth=1.2)
        ax2.plot(iters, v[:, 4], label="$\\omega_y$", linewidth=1.2)
        ax2.plot(iters, v[:, 5], label="$\\omega_z$", linewidth=1.2)
        ax2.set_xlabel("Iterations"); ax2.set_ylabel("Rotational velocities (rad/s)")
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

        label = name.replace("_", " ").title()
        fig.suptitle(f"Velocities — {label}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"velocities_{name}.pdf"), dpi=300)
        fig.savefig(os.path.join(out_dir, f"velocities_{name}.png"), dpi=200)
        plt.close(fig)

    # ─── Figure 5: 3D Trajectory with camera frustums ───
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Get start and goal from first result
    first_res = list(results_dict.values())[0]
    start_pos = first_res["poses"][0, :3, 3]

    def draw_frustum(ax, c2w, scale=0.02, color='gray', alpha=0.6):
        """Draw a small camera frustum wireframe in 3D."""
        pos = c2w[:3, 3]
        R = c2w[:3, :3]
        # Camera axes: X=right, Y=down, Z=forward (OpenCV)
        # Use short focal (wide frustum) for visual clarity
        right = R[:, 0] * scale
        down = R[:, 1] * scale * 0.7  # aspect ratio
        fwd = R[:, 2] * scale * 0.8   # short focal

        # Frustum corners (image plane)
        corners = [
            pos + fwd + right + down,
            pos + fwd - right + down,
            pos + fwd - right - down,
            pos + fwd + right - down,
        ]
        # Draw edges from pos to corners
        for c in corners:
            ax.plot([pos[0], c[0]], [pos[1], c[1]], [pos[2], c[2]],
                    color=color, linewidth=0.8, alpha=alpha)
        # Draw rectangle
        for i in range(4):
            c1, c2_ = corners[i], corners[(i + 1) % 4]
            ax.plot([c1[0], c2_[0]], [c1[1], c2_[1]], [c1[2], c2_[2]],
                    color=color, linewidth=0.8, alpha=alpha)

    for name, res in results_dict.items():
        color = colors_map.get(name, None)
        label = name.replace("_", " ").title()
        if name.startswith("inflated"):
            label = f"Inflated (α={res['sf']})"
        traj = res["poses"][:, :3, 3]
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                label=label, color=color, linewidth=1.5)

    # Frustum scale: proportional to trajectory extent
    all_traj = np.concatenate([r["poses"][:, :3, 3] for r in results_dict.values()])
    extent = np.max(np.ptp(all_traj, axis=0))
    frust_scale = extent * 0.03  # ~3% of trajectory extent

    # Draw frustums at start, goal, and intermediate poses
    # Start frustum (green)
    draw_frustum(ax, first_res["poses"][0], scale=frust_scale, color='green', alpha=0.8)
    ax.scatter(*start_pos, marker='D', color='green', s=80, zorder=10,
               edgecolors='black', linewidths=0.8)
    ax.text(start_pos[0], start_pos[1], start_pos[2], '  Initial',
            fontsize=9, fontweight='bold', color='green')

    # Goal frustum (red)
    best = min(results_dict.values(), key=lambda r: r["t_errors"][-1])
    goal_pose = best["poses"][-1]
    goal_pos = goal_pose[:3, 3]
    draw_frustum(ax, goal_pose, scale=frust_scale, color='red', alpha=0.8)
    ax.scatter(*goal_pos, marker='*', color='red', s=150, zorder=10,
               edgecolors='black', linewidths=0.8)
    ax.text(goal_pos[0], goal_pos[1], goal_pos[2], '  Desired',
            fontsize=9, fontweight='bold', color='red')

    # Intermediate frustums along one method's trajectory
    for name, res in results_dict.items():
        color = colors_map.get(name, None)
        n = len(res["poses"])
        # Draw frustums at ~25%, 50%, 75% of trajectory
        for frac in [0.25, 0.5, 0.75]:
            idx = min(int(frac * n), n - 1)
            if idx > 0 and idx < n - 1:
                draw_frustum(ax, res["poses"][idx], scale=frust_scale * 0.7,
                            color=color, alpha=0.4)

    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_zlabel("Z (m)", fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title(f"3D Camera Trajectory", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trajectory_3d.pdf"), dpi=300)
    fig.savefig(os.path.join(out_dir, "trajectory_3d.png"), dpi=200)
    plt.close(fig)

    # ─── Figure 6: Combined panel (Crombez-style) ───
    # Load saved images
    img_files = ["desired.png", "desired_inflated.png",
                 "initial.png", "initial_inflated.png"]
    imgs = {}
    for f in img_files:
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            imgs[f.replace(".png", "")] = cv2.cvtColor(
                cv2.imread(p), cv2.COLOR_BGR2RGB)

    if len(imgs) >= 4:
        fig, axes = plt.subplots(2, 4, figsize=(14, 5))

        # Row 1: desired, desired_inflated, initial, initial_inflated
        for ax_idx, (key, title) in enumerate([
            ("desired", "Desired $I^*$"),
            ("desired_inflated", f"Desired (α={list(results_dict.values())[1]['sf'] if len(results_dict) > 1 else 1.8})"),
            ("initial", "Initial $I$"),
            ("initial_inflated", f"Initial (α={list(results_dict.values())[1]['sf'] if len(results_dict) > 1 else 1.8})"),
        ]):
            if key in imgs:
                axes[0, ax_idx].imshow(imgs[key])
                axes[0, ax_idx].set_title(title, fontsize=9)
            axes[0, ax_idx].axis('off')

        # Row 2: diff desired-initial, error curve, scale param, trajectory
        if "desired" in imgs and "initial" in imgs:
            diff = np.abs(imgs["desired"].astype(float) - imgs["initial"].astype(float))
            diff = (diff / diff.max() * 255).astype(np.uint8)
            axes[1, 0].imshow(diff)
            axes[1, 0].set_title("$|I^* - I|$", fontsize=9)
            axes[1, 0].axis('off')

        # Error curves (embedded)
        for name, res in results_dict.items():
            color = colors_map.get(name, None)
            label = f"α={res['sf']}" if name.startswith("inflated") else name
            axes[1, 1].plot(res["errors"], label=label, color=color, linewidth=1.2)
        axes[1, 1].set_xlabel("Iterations", fontsize=8)
        axes[1, 1].set_ylabel("$\\|\\mathbf{e}\\|^2$", fontsize=8)
        axes[1, 1].legend(fontsize=6)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title("Residual Error", fontsize=9)

        # Scale evolution
        for name, res in results_dict.items():
            if np.all(res["scale_history"] == 1.0): continue
            color = colors_map.get(name, None)
            axes[1, 2].plot(res["scale_history"], color=color, linewidth=1.2)
        axes[1, 2].set_xlabel("Iterations", fontsize=8)
        axes[1, 2].set_ylabel("$\\alpha$", fontsize=8)
        axes[1, 2].set_title("Scale Parameter", fontsize=9)
        axes[1, 2].grid(True, alpha=0.3)

        # Mini trajectory
        ax3d = fig.add_subplot(2, 4, 8, projection='3d')
        for name, res in results_dict.items():
            color = colors_map.get(name, None)
            traj = res["poses"][:, :3, 3]
            ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                     color=color, linewidth=1.2)
        ax3d.set_title("Trajectory", fontsize=9)
        ax3d.tick_params(labelsize=6)
        axes[1, 3].remove()

        fig.suptitle(f"Experiment — Pair {pair_name}", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "combined_panel.pdf"),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, "combined_panel.png"),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)

    print(f"\n  Figures saved to {out_dir}/")


def main():
    p = argparse.ArgumentParser(description="TRO Figure Generator")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--data_factor", type=int, default=8)
    p.add_argument("--start_idx", type=int, default=94)
    p.add_argument("--goal_idx", type=int, default=60)
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--gain", type=float, default=10.0)
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--out_dir", type=str,
                   default="gs_vs_scaling_gaussians/logs/tro_figures")
    args = p.parse_args()
    device = "cuda"

    # Load scene
    cfg = load_cfg(args.cfg)
    parser = Parser(data_dir=cfg["data_dir"], factor=cfg["data_factor"],
                    normalize=cfg["normalize_world_space"], test_every=8)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    splats = ckpt["splats"]
    means = splats["means"].to(device)
    quats = splats["quats"].to(device)
    scales_orig = torch.exp(splats["scales"]).to(device)
    opacities = torch.sigmoid(splats["opacities"]).to(device)
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1).to(device)
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)

    camtoworlds = parser.camtoworlds
    W_full, H_full = list(parser.imsize_dict.values())[0]
    K_colmap = list(parser.Ks_dict.values())[0]
    df = args.data_factor
    W, H = W_full // df, H_full // df
    fx, fy = K_colmap[0, 0] / df, K_colmap[1, 1] / df
    cx, cy = K_colmap[0, 2] / df, K_colmap[1, 2] / df
    K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    cam_params = CameraParameters(px=fx, py=fy, u0=cx, v0=cy)

    c2w_start = camtoworlds[args.start_idx]
    c2w_goal = camtoworlds[args.goal_idx]
    cMo_start = np.linalg.inv(c2w_start)
    cMo_goal = np.linalg.inv(c2w_goal)

    t_init = np.linalg.norm(c2w_start[:3, 3] - c2w_goal[:3, 3])
    R_rel = c2w_start[:3, :3].T @ c2w_goal[:3, :3]
    r_init = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))

    pair = f"{args.start_idx}_{args.goal_idx}"
    out = os.path.join(args.out_dir, pair)
    os.makedirs(out, exist_ok=True)

    print(f"[TRO Figures] Pair {args.start_idx} → {args.goal_idx}")
    print(f"  Displacement: t={t_init:.3f}m, r={r_init:.1f}°")
    print(f"  Resolution: {W}x{H}, data_factor={df}")

    # Save images
    print("\nSaving images...")
    save_images(cMo_start, cMo_goal, means, quats, scales_orig, opacities,
                colors, sh_degree, K_np, W, H, 1.8, out, device)

    # Run all methods
    print("\nRunning experiments...")
    results = {}

    configs = [
        ("original", 1.0),
        ("inflated_1.8", 1.8),
        ("inflated_2.0", 2.0),
        ("coarse_to_fine", 1.8),
    ]

    for mode_name, sf in configs:
        mode = mode_name.split("_")[0] if mode_name.startswith("inflated") else mode_name
        if mode == "inflated":
            mode = "inflated"
        res = run_servo(mode, sf, c2w_start, c2w_goal,
                       means, quats, scales_orig, opacities, colors,
                       sh_degree, K_np, W, H, cam_params,
                       args.mu, args.gain, args.max_iter, device)
        results[mode_name] = res

        # Save raw data
        np.savez(os.path.join(out, f"data_{mode_name}.npz"),
                 **{k: v for k, v in res.items() if isinstance(v, np.ndarray)},
                 sf=res["sf"], n_iter=res["n_iter"],
                 converged=res["converged"])

    # Generate figures
    print("\nGenerating figures...")
    plot_experiment_panel(results, out, pair)

    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"{'Method':<20} {'Iters':>6} {'t_err(m)':>10} {'r_err(°)':>10} {'Conv':>5}")
    print("-" * 55)
    for name, res in results.items():
        print(f"{name:<20} {res['n_iter']:>6} "
              f"{res['t_errors'][-1]:>10.4f} {res['r_errors'][-1]:>10.2f} "
              f"{'Yes' if res['converged'] else 'No':>5}")


if __name__ == "__main__":
    main()
