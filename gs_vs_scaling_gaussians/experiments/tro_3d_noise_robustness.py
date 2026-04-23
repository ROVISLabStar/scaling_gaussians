"""
3D Gaussian Noise Robustness Evaluation
=========================================

Evaluates VS robustness to perturbations in the 3DGS model parameters:
  - Position noise: Gaussian noise on means μ_k
  - Scale noise: multiplicative noise on scales s_k
  - Color noise: Gaussian noise on SH coefficients
  - Opacity noise: Gaussian noise on opacities

This tests robustness to 3D reconstruction quality rather than
sensor noise — more relevant for 3DGS-based visual servoing.

Usage:
    python -m gs_vs_scaling_gaussians.experiments.tro_3d_noise_robustness \
        --ckpt <ckpt> --cfg <cfg> --goal_idx 10 --start_idx 5

Author: Youssef ALJ (UM6P)
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def perturb_gaussians(means, quats, scales_log, opacities_logit, colors,
                      noise_type, noise_level, device):
    """Apply noise to 3DGS parameters.

    Args:
        means, quats, scales_log, opacities_logit, colors: raw splat params
        noise_type: 'position', 'scale', 'color', 'opacity', 'all'
        noise_level: noise magnitude (interpretation depends on type)

    Returns:
        means', quats', scales', opacities', colors' (ready for rendering)
    """
    means_n = means.clone()
    quats_n = quats.clone()
    scales_n = torch.exp(scales_log.clone())
    opacities_n = torch.sigmoid(opacities_logit.clone())
    colors_n = colors.clone()

    if noise_type in ('position', 'all'):
        # Add Gaussian noise to positions
        # noise_level is in scene units (e.g., 0.001 = 1mm)
        means_n = means_n + noise_level * torch.randn_like(means_n)

    if noise_type in ('scale', 'all'):
        # Multiplicative noise on scales: s' = s * (1 + noise_level * N(0,1))
        scale_noise = 1.0 + noise_level * torch.randn_like(scales_n)
        scales_n = scales_n * scale_noise.clamp(min=0.1)

    if noise_type in ('color', 'all'):
        # Additive noise on SH coefficients
        colors_n = colors_n + noise_level * torch.randn_like(colors_n)

    if noise_type in ('opacity', 'all'):
        # Noise on logit-space opacities, then re-sigmoid
        op_logit_noisy = opacities_logit + noise_level * 5 * torch.randn_like(opacities_logit)
        opacities_n = torch.sigmoid(op_logit_noisy)

    return means_n, quats_n, scales_n, opacities_n, colors_n


def run_servo(c2w_start, c2w_goal, means, quats, scales, opacities, colors,
              sh_degree, K_np, W, H, cam_params, sf,
              mu=0.01, gain=10.0, max_iter=500, device="cuda"):
    """Run VS and return convergence result."""
    scales_vs = scales * sf

    cMo_goal = np.linalg.inv(c2w_goal)
    cMo_start = np.linalg.inv(c2w_start)

    gray_des, depth_des = render(cMo_goal, means, quats, scales_vs, opacities,
                                 colors, sh_degree, K_np, W, H, device)[:2]
    # Use only gray and depth
    _, gray_des, depth_des = render(cMo_goal, means, quats, scales_vs, opacities,
                                    colors, sh_degree, K_np, W, H, device)
    s_star = create_feature("pinhole", device=device, border=10)
    s_star.init(H, W)
    s_star.setCameraParameters(cam_params)
    s_star.buildFrom(gray_des, depth_des)

    wMo = np.eye(4)
    robot = SimulatorCamera()
    robot.setPosition(wMo @ np.linalg.inv(cMo_start))
    robot.setRobotState(1)
    cMo = cMo_start.copy()

    for it in range(max_iter):
        _, gray_cur, depth_cur = render(cMo, means, quats, scales_vs, opacities,
                                         colors, sh_degree, K_np, W, H, device)
        s = create_feature("pinhole", device=device, border=10)
        s.init(H, W)
        s.setCameraParameters(cam_params)
        s.buildFrom(gray_cur, depth_cur)

        error = s.error(s_star)
        c2w_cur = np.linalg.inv(cMo)
        t_err = np.linalg.norm(c2w_cur[:3, 3] - c2w_goal[:3, 3])
        R_rel = c2w_cur[:3, :3].T @ c2w_goal[:3, :3]
        r_err = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))

        if t_err < 0.01 and r_err < 1.0:
            return True, it, t_err * 1000, r_err

        err_norm = torch.sum(error ** 2).item()
        if err_norm > 1e8 or np.isnan(err_norm):
            return False, it, t_err * 1000, r_err

        Ls = s.interaction()
        Hs = Ls.T @ Ls
        diagHs = torch.diag(torch.diag(Hs))
        Hess = torch.linalg.inv(mu * diagHs + Hs + 1e-6 * torch.eye(6, device=device))
        v = -gain * (Hess @ Ls.T @ error)
        v_np = v.detach().cpu().numpy()

        vt = np.linalg.norm(v_np[:3])
        vr = np.linalg.norm(v_np[3:])
        if vt > 0.5: v_np[:3] *= 0.5 / vt
        if vr > 0.3: v_np[3:] *= 0.3 / vr

        robot.setVelocity("camera", v_np)
        wMc = robot.getPosition()
        cMo = np.linalg.inv(wMc) @ wMo

    return False, max_iter, t_err * 1000, r_err


def main():
    p = argparse.ArgumentParser(description="3D Gaussian Noise Robustness")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--data_factor", type=int, default=8)
    p.add_argument("--start_idx", type=int, default=5)
    p.add_argument("--goal_idx", type=int, default=10)
    p.add_argument("--max_iter", type=int, default=500)
    p.add_argument("--out_dir", type=str,
                   default="gs_vs_scaling_gaussians/logs/tro_3d_noise")
    args = p.parse_args()
    device = "cuda"

    os.makedirs(args.out_dir, exist_ok=True)

    # Load scene
    cfg = load_cfg(args.cfg)
    parser = Parser(data_dir=cfg["data_dir"], factor=cfg["data_factor"],
                    normalize=cfg["normalize_world_space"], test_every=8)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    splats = ckpt["splats"]

    # Keep raw params for perturbation
    means_raw = splats["means"].to(device)
    quats_raw = splats["quats"].to(device)
    scales_log = splats["scales"].to(device)      # log-space
    opacities_logit = splats["opacities"].to(device)  # logit-space
    colors_raw = torch.cat([splats["sh0"], splats["shN"]], dim=1).to(device)
    sh_degree = int(np.sqrt(colors_raw.shape[1]) - 1)

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

    # Compute scene statistics for noise calibration
    means_np = means_raw.cpu().numpy()
    scales_np = torch.exp(scales_log).cpu().numpy()
    mean_scale = np.mean(scales_np)
    print(f"[Scene] {len(means_raw)} Gaussians")
    print(f"[Scene] Mean scale: {mean_scale:.6f}")
    print(f"[Pair] {args.start_idx} -> {args.goal_idx}")

    # ─── Experiment 1: Position noise ───
    print("\n=== Position Noise ===")
    # Noise levels as fractions of mean scale
    pos_noise_levels = [0, 0.1, 0.2, 0.5, 1.0, 2.0]  # multipliers of mean_scale
    pos_noise_abs = [l * mean_scale for l in pos_noise_levels]
    scale_factors = [1.0, 2.0]

    pos_results = {}
    for nl_mult, nl_abs in zip(pos_noise_levels, pos_noise_abs):
        for sf in scale_factors:
            means_n, quats_n, scales_n, opac_n, colors_n = perturb_gaussians(
                means_raw, quats_raw, scales_log, opacities_logit, colors_raw,
                'position', nl_abs, device)
            conv, iters, t_mm, r_deg = run_servo(
                c2w_start, c2w_goal, means_n, quats_n, scales_n, opac_n, colors_n,
                sh_degree, K_np, W, H, cam_params, sf,
                max_iter=args.max_iter, device=device)
            pos_results[(nl_mult, sf)] = (conv, iters, t_mm, r_deg)
            mark = "✓" if conv else "✗"
            print(f"  σ_pos={nl_mult:.1f}×s̄ ({nl_abs:.5f}), α={sf}: "
                  f"{mark} t={t_mm:.1f}mm r={r_deg:.2f}°")

    # ─── Experiment 2: Scale noise ───
    print("\n=== Scale Noise ===")
    scale_noise_levels = [0, 0.05, 0.1, 0.2, 0.5, 1.0]  # relative std

    scale_results = {}
    for nl in scale_noise_levels:
        for sf in scale_factors:
            means_n, quats_n, scales_n, opac_n, colors_n = perturb_gaussians(
                means_raw, quats_raw, scales_log, opacities_logit, colors_raw,
                'scale', nl, device)
            conv, iters, t_mm, r_deg = run_servo(
                c2w_start, c2w_goal, means_n, quats_n, scales_n, opac_n, colors_n,
                sh_degree, K_np, W, H, cam_params, sf,
                max_iter=args.max_iter, device=device)
            scale_results[(nl, sf)] = (conv, iters, t_mm, r_deg)
            mark = "✓" if conv else "✗"
            print(f"  σ_scale={nl:.2f}, α={sf}: "
                  f"{mark} t={t_mm:.1f}mm r={r_deg:.2f}°")

    # ─── Experiment 3: Color noise ───
    print("\n=== Color (SH) Noise ===")
    color_noise_levels = [0, 0.01, 0.02, 0.05, 0.1, 0.2]

    color_results = {}
    for nl in color_noise_levels:
        for sf in scale_factors:
            means_n, quats_n, scales_n, opac_n, colors_n = perturb_gaussians(
                means_raw, quats_raw, scales_log, opacities_logit, colors_raw,
                'color', nl, device)
            conv, iters, t_mm, r_deg = run_servo(
                c2w_start, c2w_goal, means_n, quats_n, scales_n, opac_n, colors_n,
                sh_degree, K_np, W, H, cam_params, sf,
                max_iter=args.max_iter, device=device)
            color_results[(nl, sf)] = (conv, iters, t_mm, r_deg)
            mark = "✓" if conv else "✗"
            print(f"  σ_color={nl:.3f}, α={sf}: "
                  f"{mark} t={t_mm:.1f}mm r={r_deg:.2f}°")

    # ─── Render noisy scene examples ───
    print("\n=== Rendering noisy scene examples ===")
    cMo_goal = np.linalg.inv(c2w_goal)
    noise_examples = [
        ("Clean", "position", 0),
        ("Pos 0.5×s̄", "position", 0.5 * mean_scale),
        ("Pos 2.0×s̄", "position", 2.0 * mean_scale),
        ("Scale 0.2", "scale", 0.2),
        ("Scale 1.0", "scale", 1.0),
        ("Color 0.1", "color", 0.1),
    ]
    fig_ex, axes = plt.subplots(1, len(noise_examples),
                                figsize=(3 * len(noise_examples), 3))
    for i, (label, ntype, nlevel) in enumerate(noise_examples):
        means_n, quats_n, scales_n, opac_n, colors_n = perturb_gaussians(
            means_raw, quats_raw, scales_log, opacities_logit, colors_raw,
            ntype, nlevel, device)
        rgb, _, _ = render(cMo_goal, means_n, quats_n, scales_n, opac_n, colors_n,
                          sh_degree, K_np, W, H, device)
        axes[i].imshow(rgb.cpu().numpy())
        axes[i].set_title(label, fontsize=9)
        axes[i].axis('off')

    fig_ex.suptitle("Effect of 3D Gaussian perturbations on rendering",
                    fontsize=12, y=1.02)
    fig_ex.tight_layout()
    fig_ex.savefig(os.path.join(args.out_dir, "3d_noise_renders.pdf"),
                   dpi=300, bbox_inches='tight')
    fig_ex.savefig(os.path.join(args.out_dir, "3d_noise_renders.png"),
                   dpi=200, bbox_inches='tight')
    plt.close(fig_ex)

    # ─── Summary plot ───
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    COLOR_ORIG = '#d62728'
    COLOR_A20 = '#1f77b4'

    # Position noise
    ax = axes[0]
    for sf, color, label in [(1.0, COLOR_ORIG, "Original"),
                              (2.0, COLOR_A20, r"$\alpha=2.0$")]:
        t_errs = [pos_results[(nl, sf)][2] for nl in pos_noise_levels]
        conv = [pos_results[(nl, sf)][0] for nl in pos_noise_levels]
        ax.plot(pos_noise_levels, t_errs, 'o-' if sf == 1.0 else 's-',
                color=color, linewidth=1.8, markersize=6, label=label)
        # Mark failures
        for j, (nl, c) in enumerate(zip(pos_noise_levels, conv)):
            if not c:
                ax.scatter(nl, t_errs[j], marker='x', color='black',
                          s=80, zorder=10)
    ax.set_xlabel(r"Position noise ($\times \bar{s}$)", fontsize=11)
    ax.set_ylabel("Final translation error (mm)", fontsize=11)
    ax.set_title("Position noise", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Scale noise
    ax = axes[1]
    for sf, color, label in [(1.0, COLOR_ORIG, "Original"),
                              (2.0, COLOR_A20, r"$\alpha=2.0$")]:
        t_errs = [scale_results[(nl, sf)][2] for nl in scale_noise_levels]
        conv = [scale_results[(nl, sf)][0] for nl in scale_noise_levels]
        ax.plot(scale_noise_levels, t_errs, 'o-' if sf == 1.0 else 's-',
                color=color, linewidth=1.8, markersize=6, label=label)
        for j, (nl, c) in enumerate(zip(scale_noise_levels, conv)):
            if not c:
                ax.scatter(nl, t_errs[j], marker='x', color='black',
                          s=80, zorder=10)
    ax.set_xlabel(r"Scale noise $\sigma_s$", fontsize=11)
    ax.set_ylabel("Final translation error (mm)", fontsize=11)
    ax.set_title("Scale noise", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Color noise
    ax = axes[2]
    for sf, color, label in [(1.0, COLOR_ORIG, "Original"),
                              (2.0, COLOR_A20, r"$\alpha=2.0$")]:
        t_errs = [color_results[(nl, sf)][2] for nl in color_noise_levels]
        conv = [color_results[(nl, sf)][0] for nl in color_noise_levels]
        ax.plot(color_noise_levels, t_errs, 'o-' if sf == 1.0 else 's-',
                color=color, linewidth=1.8, markersize=6, label=label)
        for j, (nl, c) in enumerate(zip(color_noise_levels, conv)):
            if not c:
                ax.scatter(nl, t_errs[j], marker='x', color='black',
                          s=80, zorder=10)
    ax.set_xlabel(r"SH color noise $\sigma_c$", fontsize=11)
    ax.set_ylabel("Final translation error (mm)", fontsize=11)
    ax.set_title("Color (SH) noise", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"3D Gaussian Noise Robustness (pair {args.start_idx}→{args.goal_idx})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "3d_noise_robustness.pdf"),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(args.out_dir, "3d_noise_robustness.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ─── LaTeX table ───
    with open(os.path.join(args.out_dir, "3d_noise_table.tex"), "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Robustness to 3D Gaussian perturbations. "
                "Final positioning error (mm, $^\\circ$). "
                "\\cmark: converged. \\xmark: failed.}\n")
        f.write("\\label{tab:3d_noise}\n\\footnotesize\n")
        f.write("\\begin{tabular}{@{}lc|cc|cc@{}}\n\\toprule\n")
        f.write("& & \\multicolumn{2}{c|}{\\textbf{Original}} "
                "& \\multicolumn{2}{c}{$\\bm{\\alpha=2.0}$} \\\\\n")
        f.write("Type & Level & $\\epsilon_t$ & $\\epsilon_r$ "
                "& $\\epsilon_t$ & $\\epsilon_r$ \\\\\n\\midrule\n")

        for nl in pos_noise_levels:
            r1 = pos_results[(nl, 1.0)]
            r2 = pos_results[(nl, 2.0)]
            m1 = "\\cmark" if r1[0] else "\\xmark"
            m2 = "\\cmark" if r2[0] else "\\xmark"
            f.write(f"Pos. & {nl:.1f}$\\times\\bar{{s}}$ "
                    f"& {r1[2]:.0f}mm {m1} & {r1[3]:.1f}$^\\circ$ "
                    f"& {r2[2]:.0f}mm {m2} & {r2[3]:.1f}$^\\circ$ \\\\\n")

        f.write("\\midrule\n")
        for nl in scale_noise_levels:
            r1 = scale_results[(nl, 1.0)]
            r2 = scale_results[(nl, 2.0)]
            m1 = "\\cmark" if r1[0] else "\\xmark"
            m2 = "\\cmark" if r2[0] else "\\xmark"
            f.write(f"Scale & {nl:.2f} "
                    f"& {r1[2]:.0f}mm {m1} & {r1[3]:.1f}$^\\circ$ "
                    f"& {r2[2]:.0f}mm {m2} & {r2[3]:.1f}$^\\circ$ \\\\\n")

        f.write("\\midrule\n")
        for nl in color_noise_levels:
            r1 = color_results[(nl, 1.0)]
            r2 = color_results[(nl, 2.0)]
            m1 = "\\cmark" if r1[0] else "\\xmark"
            m2 = "\\cmark" if r2[0] else "\\xmark"
            f.write(f"Color & {nl:.2f} "
                    f"& {r1[2]:.0f}mm {m1} & {r1[3]:.1f}$^\\circ$ "
                    f"& {r2[2]:.0f}mm {m2} & {r2[3]:.1f}$^\\circ$ \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"\nAll results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
