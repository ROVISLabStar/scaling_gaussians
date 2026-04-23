"""
Scale Factor Sweep
==================

For a fixed perturbation level, sweep scale_factor from 1.0 to max_scale
and measure success rate + convergence speed for each value.
Runs in "inflated" mode (constant scale) to isolate the effect of scale magnitude.

Outputs:
  - JSON with per-scale-factor results
  - Can be plotted with plot_results.py

Usage:
    python -m gs_vs_scaling_gaussians.experiments.sweep_scale_factor \
        --ckpt <checkpoint> --cfg <config> \
        --n_trials 15 --level medium \
        --scale_min 1.0 --scale_max 5.0 --scale_steps 9 \
        --out_dir logs/scale_sweep
"""

import argparse
import json
import os
import numpy as np
import torch

from gs_vs_scaling_gaussians.experiments.run_scale_evaluation import (
    load_basic_cfg_fields, render_gsplat, se3_distance,
    perturb_pose_in_hull, build_convex_hull, run_single_vs, PERTURBATION_LEVELS,
)
from gs_vs.simulators.CameraParameters import CameraParameters
from gs_vs.datasets.colmap import Parser


def main():
    p = argparse.ArgumentParser(
        description="Sweep scale factor for inflated-mode PVS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--data_factor", type=int, default=4)
    p.add_argument("--camera_model", default="pinhole")
    p.add_argument("--feature_type", default="pinhole")

    p.add_argument("--n_trials", type=int, default=15)
    p.add_argument("--level", default="medium",
                   choices=list(PERTURBATION_LEVELS.keys()))

    p.add_argument("--scale_min", type=float, default=1.0)
    p.add_argument("--scale_max", type=float, default=5.0)
    p.add_argument("--scale_steps", type=int, default=9)

    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--lambda_", type=float, default=10.0)
    p.add_argument("--convergence_threshold", type=float, default=10000)
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out_dir", type=str, default="logs/scale_sweep")
    args = p.parse_args()
    device = "cuda"

    rng = np.random.default_rng(args.seed)

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

    preset = PERTURBATION_LEVELS[args.level]
    t_range, r_range = preset["t_range"], preset["r_range"]

    scale_factors = np.linspace(args.scale_min, args.scale_max, args.scale_steps)

    # Build convex hull
    hull = build_convex_hull(camtoworlds)
    if hull is not None:
        print(f"[Hull]  Convex hull built from {n_views} camera positions")

    print(f"[Scene] {n_views} views, {W}x{H}")
    print(f"[Sweep] {len(scale_factors)} scale factors: {scale_factors}")
    print(f"[Level] {args.level} (±{t_range:.2f}m, ±{r_range:.0f}°)")
    print(f"[Trials] {args.n_trials} per scale factor")
    print()

    # Pre-generate trial poses (same for all scale factors, inside convex hull)
    trial_poses = []
    for trial in range(args.n_trials):
        goal_idx = rng.integers(0, n_views)
        c2w_goal = camtoworlds[goal_idx]
        c2w_start = perturb_pose_in_hull(c2w_goal, t_range, r_range, rng, hull)
        trial_poses.append((goal_idx, c2w_goal, c2w_start))

    os.makedirs(args.out_dir, exist_ok=True)

    sweep_results = []

    for sf in scale_factors:
        successes = 0
        iters_list = []
        t_errs = []
        r_errs = []

        for trial, (goal_idx, c2w_goal, c2w_start) in enumerate(trial_poses):
            cMo_goal = np.linalg.inv(c2w_goal)
            cMo_start = np.linalg.inv(c2w_start)

            result = run_single_vs(
                mode="inflated",
                scale_factor=float(sf),
                cMo_start=cMo_start, cMo_goal=cMo_goal, c2w_goal=c2w_goal,
                means=means, quats=quats, scales_original=scales_original,
                opacities=opacities, colors=colors,
                sh_degree=sh_degree, K_np=K_np, W=W, H=H,
                cam_params=cam_params,
                camera_model=args.camera_model,
                feature_type=args.feature_type,
                mu=args.mu, lambda_=args.lambda_,
                max_iter=args.max_iter,
                convergence_threshold=convergence_threshold,
                device=device,
            )
            if result["converged"]:
                successes += 1
                iters_list.append(result["iterations"])
                t_errs.append(result["final_t_err"])
                r_errs.append(result["final_r_err"])

        rate = 100 * successes / args.n_trials
        avg_iter = np.mean(iters_list) if iters_list else float("nan")
        avg_t = np.mean(t_errs) if t_errs else float("nan")
        avg_r = np.mean(r_errs) if r_errs else float("nan")

        sweep_results.append({
            "scale_factor": float(sf),
            "success_rate": rate,
            "n_success": successes,
            "avg_iterations": float(avg_iter),
            "avg_t_err": float(avg_t),
            "avg_r_err": float(avg_r),
        })

        print(f"  sf={sf:.2f}: {rate:5.1f}% ({successes}/{args.n_trials}) | "
              f"avg iter={avg_iter:>6.0f} | avg t={avg_t:.4f}m r={avg_r:.2f}°")

    # Save
    out_path = os.path.join(args.out_dir, "sweep_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "level": args.level,
                "n_trials": args.n_trials,
                "scale_min": args.scale_min,
                "scale_max": args.scale_max,
                "scale_steps": args.scale_steps,
                "seed": args.seed,
            },
            "results": sweep_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
