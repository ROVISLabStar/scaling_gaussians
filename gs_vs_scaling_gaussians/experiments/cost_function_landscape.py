"""
Cost Function Landscape Visualization
=======================================

For each of the 6 DoF, sweep a single axis through the desired pose while
holding the other 5 fixed. Plot the photometric error at different scale
factors. This produces the classic "wide basin" visualization.

Directly connects to Naamani et al. IROS 2024: larger Gaussian spread
→ smoother, wider cost function basin.

Usage:
    python -m gs_vs_scaling_gaussians.experiments.cost_function_landscape \
        --ckpt <checkpoint> --cfg <config> \
        --goal_idx 0 \
        --scale_factors 1.0 1.5 2.0 3.0 \
        --t_range 0.3 --r_range 20 --n_samples 61 \
        --out_dir logs/cost_landscape
"""

import argparse
import json
import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from gs_vs.simulators.CameraParameters import CameraParameters
from gs_vs.features.factory import create_feature
from gs_vs.datasets.colmap import Parser
from gsplat.rendering import rasterization

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


DOF_LABELS = {
    "tx": r"$t_x$ (m)",
    "ty": r"$t_y$ (m)",
    "tz": r"$t_z$ (m)",
    "rx": r"$\theta_x$ (deg)",
    "ry": r"$\theta_y$ (deg)",
    "rz": r"$\theta_z$ (deg)",
}

SF_COLORS = {
    1.0: "#2196F3",
    1.2: "#03A9F4",
    1.5: "#4CAF50",
    1.8: "#FF9800",
    2.0: "#FF5722",
    2.5: "#E91E63",
    3.0: "#9C27B0",
    4.0: "#673AB7",
}


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
            elif k == "normalize_world_space": data["normalize_world_space"] = v.lower() == "true"
    data.setdefault("data_factor", 1)
    data.setdefault("normalize_world_space", True)
    return data


@torch.no_grad()
def render_gsplat(cMo, means, quats, scales, opacities, colors,
                  sh_degree, K_np, W, H, camera_model="pinhole", device="cuda"):
    viewmat = torch.from_numpy(cMo).float().to(device)[None]
    Ks = torch.from_numpy(K_np).float().to(device)[None]
    renders, _, _ = rasterization(
        means=means, quats=quats, scales=scales,
        opacities=opacities, colors=colors,
        sh_degree=sh_degree, viewmats=viewmat, Ks=Ks,
        width=W, height=H, packed=True,
        render_mode="RGB+ED", camera_model=camera_model,
    )
    rgb = torch.clamp(renders[0, ..., :3], 0.0, 1.0)
    depth = renders[0, ..., 3]
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return rgb, gray, depth


def displace_pose_dof(c2w, dof, value):
    """
    Displace c2w along a single DOF in camera frame.
    Translation DOFs: value in meters.
    Rotation DOFs: value in degrees.
    """
    c2w_d = c2w.copy()
    if dof == "tx":
        c2w_d[:3, 3] += c2w[:3, 0] * value
    elif dof == "ty":
        c2w_d[:3, 3] += c2w[:3, 1] * value
    elif dof == "tz":
        c2w_d[:3, 3] += c2w[:3, 2] * value
    elif dof in ("rx", "ry", "rz"):
        axis_idx = {"rx": 0, "ry": 1, "rz": 2}[dof]
        rotvec = np.zeros(3)
        rotvec[axis_idx] = np.radians(value)
        dR = Rot.from_rotvec(rotvec).as_matrix()
        c2w_d[:3, :3] = c2w_d[:3, :3] @ dR
    return c2w_d


def compute_cost_1d(dof, samples, c2w_goal, scale_factor,
                    means, quats, scales_original, opacities, colors,
                    sh_degree, K_np, W, H, cam_params,
                    feature_type, camera_model, device):
    """
    Sweep one DOF, compute photometric cost at each sample point.
    Both desired and current are rendered at the given scale_factor.
    """
    scales_current = scales_original * scale_factor
    cMo_goal = np.linalg.inv(c2w_goal)

    # Render desired once
    _, gray_des, depth_des = render_gsplat(
        cMo_goal, means, quats, scales_current, opacities, colors,
        sh_degree, K_np, W, H, camera_model=camera_model, device=device,
    )
    s_star = create_feature(feature_type, device=device, border=10)
    s_star.init(H, W)
    s_star.setCameraParameters(cam_params)
    s_star.buildFrom(gray_des, depth_des)

    costs = []
    for val in samples:
        c2w_cur = displace_pose_dof(c2w_goal, dof, val)
        cMo_cur = np.linalg.inv(c2w_cur)

        _, gray_cur, depth_cur = render_gsplat(
            cMo_cur, means, quats, scales_current, opacities, colors,
            sh_degree, K_np, W, H, camera_model=camera_model, device=device,
        )
        s = create_feature(feature_type, device=device, border=10)
        s.init(H, W)
        s.setCameraParameters(cam_params)
        s.buildFrom(gray_cur, depth_cur)

        error = s.error(s_star)
        cost = torch.sum(error ** 2).item()
        costs.append(cost)

    return np.array(costs)


def main():
    p = argparse.ArgumentParser(
        description="Cost function landscape visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--data_factor", type=int, default=4)
    p.add_argument("--camera_model", default="pinhole")
    p.add_argument("--feature_type", default="pinhole")

    p.add_argument("--goal_idx", type=int, default=0,
                   help="Goal view index")
    p.add_argument("--scale_factors", nargs="+", type=float,
                   default=[1.0, 1.5, 2.0, 3.0])
    p.add_argument("--dofs", nargs="+",
                   default=["tx", "ty", "tz", "rx", "ry", "rz"])

    p.add_argument("--t_range", type=float, default=0.3,
                   help="Translation sweep range (±meters)")
    p.add_argument("--r_range", type=float, default=20.0,
                   help="Rotation sweep range (±degrees)")
    p.add_argument("--n_samples", type=int, default=61,
                   help="Number of samples per sweep")

    p.add_argument("--out_dir", type=str, default="logs/cost_landscape")
    args = p.parse_args()
    device = "cuda"

    # Load scene
    cfg = load_basic_cfg_fields(args.cfg)
    parser = Parser(data_dir=cfg["data_dir"], factor=cfg["data_factor"],
                    normalize=cfg["normalize_world_space"], test_every=8)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    splats = ckpt["splats"]
    means = splats["means"].to(device)
    quats = splats["quats"].to(device)
    scales_original = torch.exp(splats["scales"]).to(device)
    opacities = torch.sigmoid(splats["opacities"]).to(device)
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1).to(device)
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)

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

    c2w_goal = camtoworlds[args.goal_idx]

    print(f"[Scene] {n_views} views, {W}x{H} (factor={df})")
    print(f"[Goal]  view {args.goal_idx}")
    print(f"[DoFs]  {args.dofs}")
    print(f"[Scales] {args.scale_factors}")
    print(f"[Range] t=±{args.t_range}m, r=±{args.r_range}°, {args.n_samples} samples")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    # Store all data for JSON export
    all_data = {}

    # Compute costs for each DOF and scale factor
    for dof in args.dofs:
        is_rotation = dof.startswith("r")
        sweep_range = args.r_range if is_rotation else args.t_range
        samples = np.linspace(-sweep_range, sweep_range, args.n_samples)

        print(f"--- {dof} ---")
        dof_data = {"samples": samples.tolist(), "costs": {}}

        for sf in args.scale_factors:
            print(f"  sf={sf:.1f} ...", end=" ", flush=True)
            costs = compute_cost_1d(
                dof, samples, c2w_goal, sf,
                means, quats, scales_original, opacities, colors,
                sh_degree, K_np, W, H, cam_params,
                args.feature_type, args.camera_model, device,
            )
            dof_data["costs"][str(sf)] = costs.tolist()
            print(f"min={costs.min():.0f} max={costs.max():.0f}")

        all_data[dof] = dof_data

    # Save raw data
    data_path = os.path.join(args.out_dir, "cost_landscape_data.json")
    with open(data_path, "w") as f:
        json.dump({
            "goal_idx": args.goal_idx,
            "scale_factors": args.scale_factors,
            "dofs": args.dofs,
            "t_range": args.t_range,
            "r_range": args.r_range,
            "data": all_data,
        }, f)
    print(f"\nData saved to {data_path}")

    # ---- Plot: 2x3 grid, one subplot per DOF ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, dof in enumerate(args.dofs[:6]):
        ax = axes[i]
        is_rotation = dof.startswith("r")
        sweep_range = args.r_range if is_rotation else args.t_range
        samples = np.array(all_data[dof]["samples"])

        for sf in args.scale_factors:
            costs = np.array(all_data[dof]["costs"][str(sf)])
            # Normalize: divide by cost at zero displacement for this sf
            mid_idx = len(samples) // 2
            cost_at_zero = costs[mid_idx] if costs[mid_idx] > 0 else 1.0
            costs_norm = costs / cost_at_zero

            color = SF_COLORS.get(sf, "#607D8B")
            ax.plot(samples, costs_norm, color=color, linewidth=1.5,
                    label=f"sf={sf:.1f}")

        ax.set_xlabel(DOF_LABELS.get(dof, dof))
        ax.set_ylabel("Normalized Cost")
        ax.set_title(dof.upper())
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.suptitle(f"Cost Function Landscape (view {args.goal_idx})", fontsize=14)
    fig.tight_layout()

    plot_path = os.path.join(args.out_dir, "cost_landscape.pdf")
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # ---- Plot: unnormalized (absolute costs) ----
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))
    axes2 = axes2.flatten()

    for i, dof in enumerate(args.dofs[:6]):
        ax = axes2[i]
        is_rotation = dof.startswith("r")
        samples = np.array(all_data[dof]["samples"])

        for sf in args.scale_factors:
            costs = np.array(all_data[dof]["costs"][str(sf)])
            color = SF_COLORS.get(sf, "#607D8B")
            ax.plot(samples, costs, color=color, linewidth=1.5,
                    label=f"sf={sf:.1f}")

        ax.set_xlabel(DOF_LABELS.get(dof, dof))
        ax.set_ylabel("Photometric Error")
        ax.set_title(dof.upper())
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    fig2.suptitle(f"Cost Function Landscape — Absolute (view {args.goal_idx})", fontsize=14)
    fig2.tight_layout()

    plot_path2 = os.path.join(args.out_dir, "cost_landscape_absolute.pdf")
    fig2.savefig(plot_path2)
    print(f"Plot saved to {plot_path2}")

    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()
