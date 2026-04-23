"""
Generate 2D cost function landscape plots (Crombez TRO Fig. 4 style).
=====================================================================

Sweeps two translation axes simultaneously and plots 3D surface.
Shows how increasing α smooths the cost function.

Usage:
    python -m gs_vs_scaling_gaussians.experiments.tro_cost_landscape \
        --ckpt <ckpt> --cfg <cfg> --goal_idx 10 \
        --out_dir gs_vs_scaling_gaussians/logs/tro_cost_landscape

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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gs_vs.simulators.CameraParameters import CameraParameters
from gs_vs.features.factory import create_feature
from gs_vs.datasets.colmap import Parser
from gs_vs.tools.SE3_tools import exponential_map
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
    return gray, depth


def compute_cost_2d(cMo_goal, means, quats, scales, opacities, colors,
                    sh_degree, K_np, W, H, cam_params,
                    axis1, axis2, range_val, n_samples, device="cuda"):
    """Sweep two DoFs and compute cost at each (x, y) point.

    Args:
        axis1, axis2: DoF indices (0=tx, 1=ty, 2=tz, 3=rx, 4=ry, 5=rz)
        range_val: sweep range (meters for translation, radians for rotation)
    """
    # Desired features
    gray_des, depth_des = render(cMo_goal, means, quats, scales, opacities,
                                 colors, sh_degree, K_np, W, H, device)
    s_star = create_feature("pinhole", device=device, border=10)
    s_star.init(H, W)
    s_star.setCameraParameters(cam_params)
    s_star.buildFrom(gray_des, depth_des)

    offsets = np.linspace(-range_val, range_val, n_samples)
    costs = np.zeros((n_samples, n_samples))

    for i, o1 in enumerate(offsets):
        for j, o2 in enumerate(offsets):
            # Perturb desired pose
            twist = np.zeros(6)
            twist[axis1] = o1
            twist[axis2] = o2
            dT = exponential_map(twist, delta_t=1.0)
            cMo_perturbed = dT @ cMo_goal

            gray_cur, depth_cur = render(cMo_perturbed, means, quats, scales,
                                         opacities, colors, sh_degree, K_np,
                                         W, H, device)
            s = create_feature("pinhole", device=device, border=10)
            s.init(H, W)
            s.setCameraParameters(cam_params)
            s.buildFrom(gray_cur, depth_cur)

            error = s.error(s_star)
            costs[i, j] = torch.sum(error ** 2).item()

        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{n_samples} rows done")

    return offsets, costs


def main():
    p = argparse.ArgumentParser(description="TRO Cost Landscape (Fig. 4 style)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--data_factor", type=int, default=8)
    p.add_argument("--goal_idx", type=int, default=10)
    p.add_argument("--t_range", type=float, default=0.3,
                   help="Translation sweep range (m)")
    p.add_argument("--n_samples", type=int, default=31,
                   help="Grid resolution per axis")
    p.add_argument("--out_dir", type=str,
                   default="gs_vs_scaling_gaussians/logs/tro_cost_landscape")
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

    cMo_goal = np.linalg.inv(camtoworlds[args.goal_idx])
    os.makedirs(args.out_dir, exist_ok=True)

    # Scale factors to compare
    scale_factors = [1.0, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]
    labels = [
        "Original ($\\alpha=1.0$)",
        "$\\alpha=1.6$",
        "$\\alpha=1.8$",
        "$\\alpha=2.0$",
        "$\\alpha=2.2$",
        "$\\alpha=2.4$",
        "$\\alpha=2.6$",
    ]

    # Sweep tx vs ty (like Crombez's u vs v translation)
    axis1, axis2 = 0, 1  # tx, ty
    axis_labels = ["$t_x$ (m)", "$t_y$ (m)"]

    print(f"[Cost Landscape] Goal idx: {args.goal_idx}")
    print(f"[Cost Landscape] Sweep: {axis_labels[0]} vs {axis_labels[1]}")
    print(f"[Cost Landscape] Range: ±{args.t_range}m, {args.n_samples}x{args.n_samples} grid")

    all_costs = {}
    offsets = None

    for sf, label in zip(scale_factors, labels):
        print(f"\n  Computing α={sf} ...")
        scales = scales_orig * sf
        offsets, costs = compute_cost_2d(
            cMo_goal, means, quats, scales, opacities, colors,
            sh_degree, K_np, W, H, cam_params,
            axis1, axis2, args.t_range, args.n_samples, device)
        all_costs[sf] = costs

    # Save raw data
    np.savez(os.path.join(args.out_dir, "cost_landscape_2d.npz"),
             offsets=offsets, **{f"costs_sf{sf}": c for sf, c in all_costs.items()},
             scale_factors=np.array(scale_factors),
             goal_idx=args.goal_idx)

    # ─── Plot: Crombez Fig. 4 style ───
    X, Y = np.meshgrid(offsets, offsets)

    # Compute global z limits for consistent colormap
    z_max = max(c.max() for c in all_costs.values())

    # Individual surface plots (one per α)
    n_plots = len(scale_factors)
    fig = plt.figure(figsize=(4 * n_plots, 4))

    for i, (sf, label) in enumerate(zip(scale_factors, labels)):
        ax = fig.add_subplot(1, n_plots, i + 1, projection='3d')
        costs = all_costs[sf]

        surf = ax.plot_surface(X, Y, costs.T, cmap='jet',
                              vmin=0, vmax=z_max,
                              edgecolor='none', alpha=0.9,
                              rstride=1, cstride=1)
        ax.set_xlabel(axis_labels[0], fontsize=8, labelpad=2)
        ax.set_ylabel(axis_labels[1], fontsize=8, labelpad=2)
        ax.set_zlabel("Cost", fontsize=8, labelpad=2)
        ax.set_title(label, fontsize=10)
        ax.set_zlim(0, z_max)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=30, azim=-60)

    fig.suptitle(f"Cost Function Landscape (goal view {args.goal_idx})",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cost_landscape_2d.pdf"),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(args.out_dir, "cost_landscape_2d.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Also plot with normalized cost per α (shows shape better)
    fig2 = plt.figure(figsize=(4 * n_plots, 4))
    for i, (sf, label) in enumerate(zip(scale_factors, labels)):
        ax = fig2.add_subplot(1, n_plots, i + 1, projection='3d')
        costs = all_costs[sf]
        costs_norm = costs / (costs.max() + 1e-8)

        surf = ax.plot_surface(X, Y, costs_norm.T, cmap='jet',
                              vmin=0, vmax=1,
                              edgecolor='none', alpha=0.9,
                              rstride=1, cstride=1)
        ax.set_xlabel(axis_labels[0], fontsize=8, labelpad=2)
        ax.set_ylabel(axis_labels[1], fontsize=8, labelpad=2)
        ax.set_zlabel("Normalized cost", fontsize=8, labelpad=2)
        ax.set_title(label, fontsize=10)
        ax.set_zlim(0, 1)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=30, azim=-60)

    fig2.suptitle(f"Normalized Cost Landscape (goal view {args.goal_idx})",
                  fontsize=12, y=1.02)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.out_dir, "cost_landscape_2d_normalized.pdf"),
                 dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(args.out_dir, "cost_landscape_2d_normalized.png"),
                 dpi=200, bbox_inches='tight')
    plt.close(fig2)

    print(f"\nFigures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
