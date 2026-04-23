"""
Convergence Domain Estimation
==============================

For each scaling mode, estimate the maximum displacement (translation and
rotation separately) from which VS still converges. Uses a bisection approach:
for a given axis (e.g., pure Z translation), binary-search the maximum
displacement that still yields convergence.

This directly connects to Naamani et al. IROS 2024 (Eq. 18): the convergence
radius is proportional to the Gaussian spread. Here we measure it empirically
for 3DGS with different scale factors.

Outputs:
  - JSON with convergence domain estimates per mode and per axis
  - Directly usable for TRO paper figures

Usage:
    python -m gs_vs_scaling_gaussians.experiments.estimate_convergence_domain \
        --ckpt <checkpoint> --cfg <config> \
        --modes original inflated coarse_to_fine error_adaptive \
        --scale_factors 1.0 1.5 2.0 3.0 \
        --out_dir logs/convergence_domain
"""

import argparse
import json
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

from gs_vs_scaling_gaussians.experiments.run_scale_evaluation import (
    load_basic_cfg_fields, se3_distance, run_single_vs,
    build_convex_hull, is_inside_hull,
)
from gs_vs.simulators.CameraParameters import CameraParameters
from gs_vs.datasets.colmap import Parser


# Axes to test: pure translations and pure rotations
DISPLACEMENT_AXES = {
    "tx": {"t": np.array([1, 0, 0], dtype=float), "r": np.zeros(3)},
    "ty": {"t": np.array([0, 1, 0], dtype=float), "r": np.zeros(3)},
    "tz": {"t": np.array([0, 0, 1], dtype=float), "r": np.zeros(3)},
    "rx": {"t": np.zeros(3), "r": np.array([1, 0, 0], dtype=float)},
    "ry": {"t": np.zeros(3), "r": np.array([0, 1, 0], dtype=float)},
    "rz": {"t": np.zeros(3), "r": np.array([0, 0, 1], dtype=float)},
}


def displace_pose(c2w, axis_def, magnitude):
    """Displace a c2w pose along a single axis by given magnitude."""
    c2w_d = c2w.copy()
    # Translation (in camera frame, transformed to world)
    dt_cam = axis_def["t"] * magnitude
    c2w_d[:3, 3] += c2w[:3, :3] @ dt_cam
    # Rotation
    dr = axis_def["r"] * np.radians(magnitude)
    if np.linalg.norm(dr) > 1e-10:
        dR = Rot.from_rotvec(dr).as_matrix()
        c2w_d[:3, :3] = c2w_d[:3, :3] @ dR
    return c2w_d


def test_convergence(magnitude, axis_def, c2w_goal, mode, scale_factor,
                     means, quats, scales_original, opacities, colors,
                     sh_degree, K_np, W, H, cam_params,
                     camera_model, feature_type, mu, lambda_,
                     max_iter, convergence_threshold, device,
                     hull=None):
    """Test if VS converges from a given displacement magnitude."""
    c2w_start = displace_pose(c2w_goal, axis_def, magnitude)

    # Reject if outside convex hull of training cameras
    if not is_inside_hull(hull, c2w_start[:3, 3]):
        return False

    cMo_start = np.linalg.inv(c2w_start)
    cMo_goal = np.linalg.inv(c2w_goal)

    result = run_single_vs(
        mode=mode, scale_factor=scale_factor,
        cMo_start=cMo_start, cMo_goal=cMo_goal, c2w_goal=c2w_goal,
        means=means, quats=quats, scales_original=scales_original,
        opacities=opacities, colors=colors,
        sh_degree=sh_degree, K_np=K_np, W=W, H=H,
        cam_params=cam_params, camera_model=camera_model,
        feature_type=feature_type, mu=mu, lambda_=lambda_,
        max_iter=max_iter, convergence_threshold=convergence_threshold,
        device=device,
    )
    return result["converged"]


def bisect_convergence(axis_name, axis_def, c2w_goal, mode, scale_factor,
                       lo, hi, n_bisect, n_confirm,
                       means, quats, scales_original, opacities, colors,
                       sh_degree, K_np, W, H, cam_params,
                       camera_model, feature_type, mu, lambda_,
                       max_iter, convergence_threshold, device,
                       hull=None):
    """
    Binary search for the maximum displacement that still converges.
    At each bisection step, test n_confirm times (from both +/- directions)
    and require majority success.
    """
    for step in range(n_bisect):
        mid = (lo + hi) / 2
        # Test from both positive and negative displacement
        successes = 0
        for sign in [1.0, -1.0]:
            for _ in range(n_confirm):
                ok = test_convergence(
                    sign * mid, axis_def, c2w_goal, mode, scale_factor,
                    means, quats, scales_original, opacities, colors,
                    sh_degree, K_np, W, H, cam_params,
                    camera_model, feature_type, mu, lambda_,
                    max_iter, convergence_threshold, device,
                    hull=hull,
                )
                if ok:
                    successes += 1

        total_tests = 2 * n_confirm
        if successes >= total_tests // 2:
            lo = mid  # Still converges — try larger
        else:
            hi = mid  # Fails — try smaller

    return (lo + hi) / 2


def main():
    p = argparse.ArgumentParser(
        description="Estimate convergence domain for each scaling mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--data_factor", type=int, default=4)
    p.add_argument("--camera_model", default="pinhole")
    p.add_argument("--feature_type", default="pinhole")

    p.add_argument("--goal_idx", type=int, default=None,
                   help="Goal view index (default: middle of dataset)")
    p.add_argument("--modes", nargs="+",
                   default=["original", "inflated", "coarse_to_fine", "error_adaptive"])
    p.add_argument("--scale_factors", nargs="+", type=float, default=[1.0, 1.5, 2.0, 3.0],
                   help="Scale factors to test (for inflated mode, each is tested separately)")
    p.add_argument("--axes", nargs="+", default=["tx", "ty", "tz", "rx", "ry", "rz"],
                   choices=list(DISPLACEMENT_AXES.keys()))

    # Bisection parameters
    p.add_argument("--t_max", type=float, default=0.5,
                   help="Max translation to search (meters)")
    p.add_argument("--r_max", type=float, default=40.0,
                   help="Max rotation to search (degrees)")
    p.add_argument("--n_bisect", type=int, default=8,
                   help="Number of bisection steps")
    p.add_argument("--n_confirm", type=int, default=1,
                   help="Confirmations per direction per step")

    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--lambda_", type=float, default=10.0)
    p.add_argument("--convergence_threshold", type=float, default=10000)
    p.add_argument("--max_iter", type=int, default=2000)

    p.add_argument("--out_dir", type=str, default="logs/convergence_domain")
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

    pixel_ratio = (W * H) / (W_full * H_full)
    convergence_threshold = args.convergence_threshold * pixel_ratio

    goal_idx = args.goal_idx if args.goal_idx is not None else n_views // 2
    c2w_goal = camtoworlds[goal_idx]

    # Build convex hull
    hull = build_convex_hull(camtoworlds)
    if hull is not None:
        print(f"[Hull]  Convex hull built from {n_views} camera positions")

    print(f"[Scene] {n_views} views, {W}x{H}")
    print(f"[Goal]  view {goal_idx}")
    print(f"[Axes]  {args.axes}")
    print(f"[Bisect] {args.n_bisect} steps, {args.n_confirm} confirms")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    # For "inflated" mode, test each scale_factor separately.
    # For other modes, use the first scale_factor in the list.
    results = []

    for mode in args.modes:
        sf_list = args.scale_factors if mode == "inflated" else [args.scale_factors[-1]]

        for sf in sf_list:
            # For "original" mode, force sf=1.0
            if mode == "original":
                sf = 1.0

            label = f"{mode}" if mode != "inflated" else f"inflated(sf={sf:.1f})"
            print(f"--- {label} ---")

            mode_result = {"mode": mode, "scale_factor": sf, "axes": {}}

            for axis_name in args.axes:
                axis_def = DISPLACEMENT_AXES[axis_name]
                is_rotation = np.linalg.norm(axis_def["r"]) > 0
                hi = args.r_max if is_rotation else args.t_max

                max_disp = bisect_convergence(
                    axis_name, axis_def, c2w_goal, mode, sf,
                    lo=0.0, hi=hi, n_bisect=args.n_bisect,
                    n_confirm=args.n_confirm,
                    means=means, quats=quats, scales_original=scales_original,
                    opacities=opacities, colors=colors,
                    sh_degree=sh_degree, K_np=K_np, W=W, H=H,
                    cam_params=cam_params, camera_model=args.camera_model,
                    feature_type=args.feature_type,
                    mu=args.mu, lambda_=args.lambda_,
                    max_iter=args.max_iter,
                    convergence_threshold=convergence_threshold,
                    device=device,
                    hull=hull,
                )
                unit = "°" if is_rotation else "m"
                print(f"  {axis_name}: ±{max_disp:.4f}{unit}")
                mode_result["axes"][axis_name] = float(max_disp)

            results.append(mode_result)

            # Skip redundant sf iterations for non-inflated modes
            if mode == "original":
                break

        print()

    # Save
    out_path = os.path.join(args.out_dir, "convergence_domain.json")
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "goal_idx": goal_idx,
                "n_bisect": args.n_bisect,
                "n_confirm": args.n_confirm,
                "t_max": args.t_max,
                "r_max": args.r_max,
                "max_iter": args.max_iter,
            },
            "results": results,
        }, f, indent=2)
    print(f"Saved to {out_path}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"Convergence Domain Summary (goal view {goal_idx})")
    print(f"{'='*70}")
    header = f"{'Mode':>25}"
    for ax in args.axes:
        unit = "°" if np.linalg.norm(DISPLACEMENT_AXES[ax]["r"]) > 0 else "m"
        header += f" | {ax+'('+unit+')':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        label = r["mode"] if r["mode"] != "inflated" else f"inflated(sf={r['scale_factor']:.1f})"
        row = f"{label:>25}"
        for ax in args.axes:
            row += f" | {r['axes'].get(ax, 0):>8.3f}"
        print(row)
    print()


if __name__ == "__main__":
    main()
