"""
TRO Extra Figures — Scale-Adaptive PVS with 3DGS.
==================================================

Generates three sets of publication-quality figures in the style of
Crombez et al. TRO 2019:

  1. Initial images mosaic (Fig. 12 style): desired image + 20 initial views
  2. Challenging cases (Fig. 13 style): pairs where original fails but α=2.0 converges
  3. Noise robustness (Table I style): final errors vs noise level σ

Usage:
    python -m gs_vs_scaling_gaussians.experiments.tro_extra_figures \
        --ckpt <ckpt> --cfg <cfg> --data_factor 8 --goal_idx 10 \
        --out_dir gs_vs_scaling_gaussians/logs/tro_extra_figures

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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gs_vs.simulators.RobotSimulator import SimulatorCamera
from gs_vs.simulators.CameraParameters import CameraParameters
from gs_vs.features.factory import create_feature
from gs_vs.datasets.colmap import Parser
from gsplat.rendering import rasterization

# ─────────────────────────────────────────────────────────────────────────────
# Path to pre-computed convergence table results
CONVERGENCE_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "papier_TRO_scaling_gaussians",
    "gs_vs_scaling_gaussians",
    "logs",
    "tro_convergence_table_v3",
    "convergence_results.json",
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities

def load_cfg(path):
    data = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if k == "data_dir":
                data["data_dir"] = v
            elif k == "data_factor":
                data["data_factor"] = int(v)
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


def run_servo(sf, c2w_start, c2w_goal,
              means, quats, scales_orig, opacities, colors,
              sh_degree, K_np, W, H, cam_params,
              mu=0.01, gain=10.0, max_iter=500,
              noise_sigma=0.0, device="cuda"):
    """
    Run visual servoing with scale factor sf.
    Returns dict with convergence info and trajectory.
    noise_sigma: std of Gaussian noise added to rendered images (0=no noise).
    """
    scales = scales_orig * sf

    cMo_goal = np.linalg.inv(c2w_goal)
    cMo_start = np.linalg.inv(c2w_start)

    # Desired image
    rgb_des, gray_des, depth_des = render(
        cMo_goal, means, quats, scales, opacities, colors,
        sh_degree, K_np, W, H, device)
    if noise_sigma > 0.0:
        gray_des = gray_des + noise_sigma * torch.randn_like(gray_des)
        gray_des = gray_des.clamp(0.0, 1.0)

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

    errors = []
    poses = []
    t_errors = []
    r_errors = []

    for it in range(max_iter):
        rgb_cur, gray_cur, depth_cur = render(
            cMo, means, quats, scales, opacities, colors,
            sh_degree, K_np, W, H, device)
        if noise_sigma > 0.0:
            gray_cur = gray_cur + noise_sigma * torch.randn_like(gray_cur)
            gray_cur = gray_cur.clamp(0.0, 1.0)

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

        if t_err < 0.01 and r_err < 1.0:
            break

        # LM update
        Ls = s.interaction()
        Hs = Ls.T @ Ls
        diagHs = torch.diag(torch.diag(Hs))
        Hess = torch.linalg.inv(mu * diagHs + Hs + 1e-6 * torch.eye(6, device=device))
        v = -gain * (Hess @ Ls.T @ error)
        v_np = v.detach().cpu().numpy()

        # Velocity clamping for stability
        vt = np.linalg.norm(v_np[:3])
        vr = np.linalg.norm(v_np[3:])
        if vt > 0.5:
            v_np[:3] *= 0.5 / vt
        if vr > 0.3:
            v_np[3:] *= 0.3 / vr

        robot.setVelocity("camera", v_np)
        wMc = robot.getPosition()
        cMo = np.linalg.inv(wMc) @ wMo

    converged = t_errors[-1] < 0.01 and r_errors[-1] < 1.0
    print(f"    [sf={sf:.1f}, noise={noise_sigma:.3f}] "
          f"iters={len(errors)}, t={t_errors[-1]:.4f}m, "
          f"r={r_errors[-1]:.2f}°, conv={'Y' if converged else 'N'}")

    return {
        "sf": sf,
        "errors": np.array(errors),
        "poses": np.array(poses),
        "t_errors": np.array(t_errors),
        "r_errors": np.array(r_errors),
        "converged": converged,
        "n_iter": len(errors),
    }


def save_fig(fig, out_dir, name):
    """Save figure as both PDF (300 dpi) and PNG (200 dpi)."""
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {name}.pdf/.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Initial images mosaic

def figure1_initial_mosaic(conv_data, camtoworlds, goal_idx,
                            means, quats, scales_orig, opacities, colors,
                            sh_degree, K_np, W, H, out_dir, device="cuda"):
    """
    Render desired image (goal) and all 20 start images.
    Layout: 1 desired image (full row) + 4x5 grid of initial images.
    Style: Crombez TRO 2019 Fig. 12.
    """
    print("\n[Figure 1] Rendering initial images mosaic...")
    start_indices = conv_data["start_indices"]

    c2w_goal = camtoworlds[goal_idx]
    cMo_goal = np.linalg.inv(c2w_goal)

    # Render desired
    rgb_des, _, _ = render(cMo_goal, means, quats, scales_orig, opacities,
                           colors, sh_degree, K_np, W, H, device)
    img_des = (rgb_des.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    # Render all start images
    start_imgs = []
    for idx in start_indices:
        c2w = camtoworlds[idx]
        cMo = np.linalg.inv(c2w)
        rgb, _, _ = render(cMo, means, quats, scales_orig, opacities,
                           colors, sh_degree, K_np, W, H, device)
        img = (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        start_imgs.append(img)
        print(f"    Rendered start idx={idx}")

    n = len(start_imgs)  # 20
    ncols = 5
    nrows = n // ncols  # 4

    # ── Layout: top row = desired image, bottom = 4x5 grid ──
    fig = plt.figure(figsize=(ncols * 2.5, (nrows + 1.2) * 2.5))

    # Top row: desired image centered
    ax_des = fig.add_axes([0.2, 0.78, 0.6, 0.2])
    ax_des.imshow(img_des)
    ax_des.set_title(r"Desired image $I^*$ (goal index 10)", fontsize=13, fontweight='bold')
    ax_des.axis('off')

    # Grid: 4 rows x 5 cols of initial images
    for r in range(nrows):
        for c in range(ncols):
            i = r * ncols + c
            left = c / ncols
            # start at 0.78 - small gap, then 4 rows below
            bottom = 0.75 - (r + 1) * (0.72 / nrows)
            width = 0.9 / ncols
            height = 0.68 / nrows
            ax = fig.add_axes([left + 0.04, bottom, width, height])
            ax.imshow(start_imgs[i])
            ax.set_title(f"$I_{{0}}$ (idx {start_indices[i]})",
                         fontsize=7, pad=2)
            ax.axis('off')

    fig.suptitle("Initial configurations used in the convergence study",
                 fontsize=12, y=1.01)
    save_fig(fig, out_dir, "initial_images_mosaic")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Challenging cases

def figure2_challenging_cases(conv_data, camtoworlds, goal_idx,
                               means, quats, scales_orig, opacities, colors,
                               sh_degree, K_np, W, H, cam_params,
                               out_dir, device="cuda"):
    """
    Pairs where Original fails but alpha=2.0 converges.
    For each pair: show desired / initial / difference + 3D trajectory.
    Style: Crombez TRO 2019 Fig. 13.
    """
    print("\n[Figure 2] Challenging cases (Original fails, alpha=2.0 converges)...")

    results_json = conv_data["results"]

    # Select pairs: Original=fail AND alpha=2.0=converge
    challenging = []
    for idx in conv_data["start_indices"]:
        k_orig = f"{idx}_Original"
        k_a20 = f"{idx}_\u03b1=2.0"
        if k_orig in results_json and k_a20 in results_json:
            if not results_json[k_orig]["converged"] and results_json[k_a20]["converged"]:
                challenging.append(idx)

    print(f"  Found {len(challenging)} challenging pairs: {challenging}")
    # Use up to 3 pairs for a clean figure
    challenging = challenging[:3]

    n_pairs = len(challenging)
    c2w_goal = camtoworlds[goal_idx]
    cMo_goal = np.linalg.inv(c2w_goal)
    rgb_des, _, _ = render(cMo_goal, means, quats, scales_orig, opacities,
                           colors, sh_degree, K_np, W, H, device)
    img_des = (rgb_des.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    # Colors for methods
    COLOR_ORIG = "#d62728"
    COLOR_A20 = "#1f77b4"

    # One row per pair, columns: desired | initial | diff | trajectory
    # Plus a shared "Desired image" column header note
    fig = plt.figure(figsize=(14, 3.5 * n_pairs))
    gs_outer = gridspec.GridSpec(n_pairs, 4, figure=fig,
                                 hspace=0.35, wspace=0.08)

    pair_results = {}  # store for later inspection

    for row, start_idx in enumerate(challenging):
        print(f"\n  Running VS for pair {start_idx} -> {goal_idx}...")
        c2w_start = camtoworlds[start_idx]
        cMo_start = np.linalg.inv(c2w_start)

        rgb_init, _, _ = render(cMo_start, means, quats, scales_orig, opacities,
                                colors, sh_degree, K_np, W, H, device)
        img_init = (rgb_init.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # Initial displacement
        t_init = np.linalg.norm(c2w_start[:3, 3] - c2w_goal[:3, 3])
        R_rel = c2w_start[:3, :3].T @ c2w_goal[:3, :3]
        r_init = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))

        # Difference image
        diff = np.abs(img_des.astype(np.float32) - img_init.astype(np.float32))
        diff_disp = (diff / (diff.max() + 1e-6) * 255).astype(np.uint8)
        diff_colored = cv2.applyColorMap(diff_disp, cv2.COLORMAP_JET)
        diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)

        # Run VS: original and alpha=2.0
        print(f"    Running original (sf=1.0)...")
        res_orig = run_servo(
            1.0, c2w_start, c2w_goal,
            means, quats, scales_orig, opacities, colors,
            sh_degree, K_np, W, H, cam_params,
            mu=0.01, gain=10.0, max_iter=500, device=device)

        print(f"    Running alpha=2.0...")
        res_a20 = run_servo(
            2.0, c2w_start, c2w_goal,
            means, quats, scales_orig, opacities, colors,
            sh_degree, K_np, W, H, cam_params,
            mu=0.01, gain=10.0, max_iter=500, device=device)

        pair_results[start_idx] = {"original": res_orig, "alpha_2.0": res_a20}

        # ── Col 0: Desired image ──
        ax0 = fig.add_subplot(gs_outer[row, 0])
        ax0.imshow(img_des)
        if row == 0:
            ax0.set_title(r"Desired $I^*$", fontsize=10, fontweight='bold')
        ax0.set_ylabel(f"Pair {start_idx}→{goal_idx}\n"
                       f"$\\Delta t$={t_init:.2f}m, $\\Delta\\theta$={r_init:.1f}°",
                       fontsize=8)
        ax0.axis('off')

        # ── Col 1: Initial image ──
        ax1 = fig.add_subplot(gs_outer[row, 1])
        ax1.imshow(img_init)
        if row == 0:
            ax1.set_title(r"Initial $I_0$", fontsize=10, fontweight='bold')
        ax1.axis('off')

        # ── Col 2: Difference image ──
        ax2 = fig.add_subplot(gs_outer[row, 2])
        ax2.imshow(diff_colored)
        if row == 0:
            ax2.set_title(r"$|I^* - I_0|$", fontsize=10, fontweight='bold')
        ax2.axis('off')

        # ── Col 3: 3D trajectory ──
        ax3 = fig.add_subplot(gs_outer[row, 3], projection='3d')

        traj_orig = res_orig["poses"][:, :3, 3]
        traj_a20 = res_a20["poses"][:, :3, 3]
        goal_pos = c2w_goal[:3, 3]
        start_pos = c2w_start[:3, 3]

        ax3.plot(traj_orig[:, 0], traj_orig[:, 1], traj_orig[:, 2],
                 color=COLOR_ORIG, linewidth=1.2, label="Original")
        ax3.plot(traj_a20[:, 0], traj_a20[:, 1], traj_a20[:, 2],
                 color=COLOR_A20, linewidth=1.2, label=r"$\alpha=2.0$")

        ax3.scatter(*start_pos, marker='D', color='green', s=40, zorder=5)
        ax3.scatter(*goal_pos, marker='*', color='red', s=80, zorder=5)

        if row == 0:
            ax3.set_title("3D Trajectory", fontsize=10, fontweight='bold')
        ax3.tick_params(labelsize=5)
        ax3.set_xlabel("X", fontsize=6)
        ax3.set_ylabel("Y", fontsize=6)
        ax3.set_zlabel("Z", fontsize=6)
        ax3.legend(fontsize=6, loc='upper left')

    # Convergence status annotation
    status_lines = []
    for start_idx in challenging:
        r_orig = pair_results[start_idx]["original"]
        r_a20 = pair_results[start_idx]["alpha_2.0"]
        sym_orig = "OK" if r_orig["converged"] else "FAIL"
        sym_a20 = "OK" if r_a20["converged"] else "FAIL"
        status_lines.append(
            f"Pair {start_idx}: Original={sym_orig}, alpha=2.0={sym_a20}"
        )

    fig.suptitle(
        "Challenging configurations: Original PVS fails, Scale-Adaptive PVS ($\\alpha=2.0$) converges",
        fontsize=11, y=1.01)
    save_fig(fig, out_dir, "challenging_cases")

    # Save trajectory data
    for start_idx in challenging:
        np.savez(os.path.join(out_dir, f"challenging_pair_{start_idx}.npz"),
                 traj_orig=pair_results[start_idx]["original"]["poses"],
                 traj_a20=pair_results[start_idx]["alpha_2.0"]["poses"],
                 errors_orig=pair_results[start_idx]["original"]["errors"],
                 errors_a20=pair_results[start_idx]["alpha_2.0"]["errors"])

    return pair_results


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Noise robustness

def figure3_noise_robustness(camtoworlds, start_idx, goal_idx,
                              means, quats, scales_orig, opacities, colors,
                              sh_degree, K_np, W, H, cam_params,
                              out_dir, device="cuda"):
    """
    Run VS at increasing noise levels with original and alpha=2.0.
    Generate:
      (a) Figure showing noisy desired images
      (b) Table of final errors (LaTeX)
    Style: Crombez TRO 2019 Table I.
    """
    print(f"\n[Figure 3] Noise robustness (pair {start_idx} -> {goal_idx})...")

    # Noise levels on [0,255] scale, converted to [0,1]
    sigma_levels = [0.0, 5/255, 10/255, 20/255, 40/255]
    scale_factors = [1.0, 2.0]

    c2w_start = camtoworlds[start_idx]
    c2w_goal = camtoworlds[goal_idx]
    cMo_goal = np.linalg.inv(c2w_goal)

    # ── (a) Noisy image visualization ──
    print("  Rendering noisy images...")
    rgb_clean, _, _ = render(cMo_goal, means, quats, scales_orig, opacities,
                             colors, sh_degree, K_np, W, H, device)

    fig_noise, axes = plt.subplots(1, len(sigma_levels), figsize=(3 * len(sigma_levels), 3.2))
    for j, sigma in enumerate(sigma_levels):
        if sigma == 0.0:
            noisy = rgb_clean.cpu().numpy()
        else:
            noise = sigma * torch.randn_like(rgb_clean)
            noisy = (rgb_clean + noise).clamp(0.0, 1.0).cpu().numpy()
        axes[j].imshow(noisy)
        sigma_255 = sigma * 255
        axes[j].set_title(f"$\\sigma={sigma_255:.1f}$", fontsize=10)
        axes[j].axis('off')

    fig_noise.suptitle("Desired image $I^*$ with increasing Gaussian noise",
                       fontsize=11)
    fig_noise.tight_layout()
    save_fig(fig_noise, out_dir, "noisy_images")

    # ── (b) Run VS for each sigma x scale_factor ──
    print("  Running VS experiments...")
    # Table: rows=sigma, cols=sf; each cell: t_err(mm), r_err(deg), converged
    table_data = {}  # (sigma, sf) -> dict

    for sigma in sigma_levels:
        for sf in scale_factors:
            print(f"  sigma={sigma:.3f}, sf={sf:.1f}")
            res = run_servo(
                sf, c2w_start, c2w_goal,
                means, quats, scales_orig, opacities, colors,
                sh_degree, K_np, W, H, cam_params,
                mu=0.01, gain=10.0, max_iter=500,
                noise_sigma=sigma, device=device)
            table_data[(sigma, sf)] = res

    # ── Error vs sigma curves ──
    fig_curves, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    COLOR_ORIG = "#d62728"
    COLOR_A20 = "#1f77b4"

    t_orig = [table_data[(s, 1.0)]["t_errors"][-1] * 1000 for s in sigma_levels]
    r_orig = [table_data[(s, 1.0)]["r_errors"][-1] for s in sigma_levels]
    t_a20 = [table_data[(s, 2.0)]["t_errors"][-1] * 1000 for s in sigma_levels]
    r_a20 = [table_data[(s, 2.0)]["r_errors"][-1] for s in sigma_levels]

    sigma_arr = np.array(sigma_levels) * 255  # display on [0,255] scale
    ax1.plot(sigma_arr, t_orig, 'o-', color=COLOR_ORIG, linewidth=1.8,
             markersize=6, label="Original")
    ax1.plot(sigma_arr, t_a20, 's-', color=COLOR_A20, linewidth=1.8,
             markersize=6, label=r"$\alpha=2.0$")
    ax1.set_xlabel("Noise $\\sigma$", fontsize=11)
    ax1.set_ylabel("Final translation error (mm)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Translation error vs. noise", fontsize=11)

    ax2.plot(sigma_arr, r_orig, 'o-', color=COLOR_ORIG, linewidth=1.8,
             markersize=6, label="Original")
    ax2.plot(sigma_arr, r_a20, 's-', color=COLOR_A20, linewidth=1.8,
             markersize=6, label=r"$\alpha=2.0$")
    ax2.set_xlabel("Noise $\\sigma$", fontsize=11)
    ax2.set_ylabel("Final rotation error (°)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Rotation error vs. noise", fontsize=11)

    fig_curves.suptitle(f"Noise robustness (pair {start_idx}→{goal_idx})", fontsize=12)
    fig_curves.tight_layout()
    save_fig(fig_curves, out_dir, "noise_robustness")

    # ── LaTeX table ──
    tex_path = os.path.join(out_dir, "noise_table.tex")
    with open(tex_path, "w") as f:
        f.write("% Noise robustness table — Scale-Adaptive PVS\n")
        f.write("% t_err in mm, r_err in degrees\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Final pose error vs. image noise level. "
                "Convergence criterion: $\\epsilon_t < 10\\,$mm and "
                "$\\epsilon_r < 1^\\circ$.}\n")
        f.write("\\label{tab:noise_robustness}\n")
        f.write("\\begin{tabular}{c|cc|cc}\n")
        f.write("\\toprule\n")
        f.write("& \\multicolumn{2}{c|}{Original ($\\alpha=1.0$)} "
                "& \\multicolumn{2}{c}{Scale-adaptive ($\\alpha=2.0$)} \\\\\n")
        f.write("$\\sigma$ & $\\epsilon_t$ (mm) & $\\epsilon_r$ (\\degree) "
                "& $\\epsilon_t$ (mm) & $\\epsilon_r$ (\\degree) \\\\\n")
        f.write("\\midrule\n")
        for sigma in sigma_levels:
            r1 = table_data[(sigma, 1.0)]
            r2 = table_data[(sigma, 2.0)]
            t1 = r1["t_errors"][-1] * 1000
            ro1 = r1["r_errors"][-1]
            t2 = r2["t_errors"][-1] * 1000
            ro2 = r2["r_errors"][-1]
            conv1 = "\\cmark" if r1["converged"] else "\\xmark"
            conv2 = "\\cmark" if r2["converged"] else "\\xmark"
            sigma_255 = sigma * 255
            f.write(f"{sigma_255:.1f} & {t1:.1f} {conv1} & {ro1:.2f} {conv1} "
                    f"& {t2:.1f} {conv2} & {ro2:.2f} {conv2} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  Saved noise_table.tex")

    # Print summary to console
    print("\n  Noise robustness summary:")
    print(f"  {'sigma':>6} | {'t_orig(mm)':>12} {'r_orig(°)':>10} {'conv_orig':>10} "
          f"| {'t_a20(mm)':>12} {'r_a20(°)':>10} {'conv_a20':>10}")
    print("  " + "-" * 80)
    for sigma in sigma_levels:
        r1 = table_data[(sigma, 1.0)]
        r2 = table_data[(sigma, 2.0)]
        print(f"  {sigma:>6.3f} | "
              f"{r1['t_errors'][-1]*1000:>12.1f} {r1['r_errors'][-1]:>10.2f} "
              f"{'Y' if r1['converged'] else 'N':>10} | "
              f"{r2['t_errors'][-1]*1000:>12.1f} {r2['r_errors'][-1]:>10.2f} "
              f"{'Y' if r2['converged'] else 'N':>10}")

    return table_data


# ─────────────────────────────────────────────────────────────────────────────
# Main

def main():
    p = argparse.ArgumentParser(description="TRO Extra Figures Generator")
    p.add_argument("--ckpt", required=True, help="Path to gsplat checkpoint (.pt)")
    p.add_argument("--cfg", required=True, help="Path to gsplat config (.yml)")
    p.add_argument("--data_factor", type=int, default=8,
                   help="Image downscale factor (default: 8)")
    p.add_argument("--goal_idx", type=int, default=10,
                   help="Goal camera index (default: 10)")
    p.add_argument("--noise_start_idx", type=int, default=5,
                   help="Start idx for noise robustness experiment (default: 5)")
    p.add_argument("--out_dir", type=str,
                   default="gs_vs_scaling_gaussians/logs/tro_extra_figures",
                   help="Output directory")
    p.add_argument("--figures", type=str, default="1,2,3",
                   help="Comma-separated list of figures to generate (1,2,3)")
    args = p.parse_args()

    figures_to_run = set(args.figures.split(","))
    device = "cuda"

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[TRO Extra Figures] Output: {args.out_dir}")
    print(f"  data_factor={args.data_factor}, goal_idx={args.goal_idx}")

    # ── Load scene ──
    print("\nLoading scene...")
    cfg = load_cfg(args.cfg)
    parser = Parser(
        data_dir=cfg["data_dir"],
        factor=cfg["data_factor"],
        normalize=cfg["normalize_world_space"],
        test_every=8,
    )
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    splats = ckpt["splats"]
    means = splats["means"].to(device)
    quats = splats["quats"].to(device)
    scales_orig = torch.exp(splats["scales"]).to(device)
    opacities = torch.sigmoid(splats["opacities"]).to(device)
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1).to(device)
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)
    print(f"  Loaded {means.shape[0]:,} Gaussians, sh_degree={sh_degree}")

    # ── Camera setup at requested data_factor ──
    camtoworlds = parser.camtoworlds
    W_full, H_full = list(parser.imsize_dict.values())[0]
    K_colmap = list(parser.Ks_dict.values())[0]
    df = args.data_factor
    W, H = W_full // df, H_full // df
    fx = K_colmap[0, 0] / df
    fy = K_colmap[1, 1] / df
    cx = K_colmap[0, 2] / df
    cy = K_colmap[1, 2] / df
    K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    cam_params = CameraParameters(px=fx, py=fy, u0=cx, v0=cy)
    print(f"  Resolution: {W}x{H}, fx={fx:.1f}, fy={fy:.1f}")
    print(f"  Dataset: {len(camtoworlds)} cameras")

    # ── Load convergence JSON ──
    print(f"\nLoading convergence data from:\n  {CONVERGENCE_JSON}")
    with open(CONVERGENCE_JSON) as f:
        conv_data = json.load(f)
    print(f"  goal_idx={conv_data['goal_idx']}, "
          f"n_starts={len(conv_data['start_indices'])}")

    # ── Figure 1: Initial images mosaic ──
    if "1" in figures_to_run:
        figure1_initial_mosaic(
            conv_data, camtoworlds, args.goal_idx,
            means, quats, scales_orig, opacities, colors,
            sh_degree, K_np, W, H, args.out_dir, device)

    # ── Figure 2: Challenging cases ──
    if "2" in figures_to_run:
        figure2_challenging_cases(
            conv_data, camtoworlds, args.goal_idx,
            means, quats, scales_orig, opacities, colors,
            sh_degree, K_np, W, H, cam_params,
            args.out_dir, device)

    # ── Figure 3: Noise robustness ──
    if "3" in figures_to_run:
        figure3_noise_robustness(
            camtoworlds, args.noise_start_idx, args.goal_idx,
            means, quats, scales_orig, opacities, colors,
            sh_degree, K_np, W, H, cam_params,
            args.out_dir, device)

    print(f"\n[Done] All figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
