"""
Batch Evaluation: Scale-Adaptive PVS in 3DGS
=============================================

Systematic comparison of scaling modes across multiple perturbation levels.
For each trial: pick a random goal view, apply a random SE(3) perturbation,
run VS with each mode, and record success/failure + metrics.

Perturbation levels:
  SMALL:  ±0.05m translation, ±5° rotation
  MEDIUM: ±0.15m translation, ±15° rotation
  LARGE:  ±0.30m translation, ±30° rotation

Outputs:
  - JSON results file with per-trial metrics
  - Summary statistics printed to stdout

Usage:
    python -m gs_vs_scaling_gaussians.experiments.run_scale_evaluation \
        --ckpt <checkpoint> --cfg <config> \
        --n_trials 20 --levels small medium large \
        --modes original inflated coarse_to_fine error_adaptive \
        --out_dir logs/scale_evaluation
"""

import argparse
import json
import os
import time
import torch
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import ConvexHull, Delaunay

from gs_vs.simulators.RobotSimulator import SimulatorCamera
from gs_vs.simulators.CameraParameters import CameraParameters
from gs_vs.features.factory import create_feature
from gs_vs.datasets.colmap import Parser
from gsplat.rendering import rasterization


# ============================================================
# Perturbation presets
# ============================================================
PERTURBATION_LEVELS = {
    "small":  {"t_range": 0.05, "r_range": 5.0},
    "medium": {"t_range": 0.15, "r_range": 15.0},
    "large":  {"t_range": 0.30, "r_range": 30.0},
}

MODES_DEFAULT = ["original", "inflated", "coarse_to_fine", "smooth_decay", "error_adaptive"]


# ============================================================
# Shared helpers (same as scale_adaptive_vs.py)
# ============================================================
def se3_distance(c2w_a, c2w_b):
    t_dist = np.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3])
    R_rel = c2w_a[:3, :3].T @ c2w_b[:3, :3]
    angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
    return t_dist, np.degrees(angle)


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


def compute_scale_schedule(mode, scale_factor, max_iter, n_phases=5):
    if mode == "original":
        return np.ones(max_iter)
    elif mode == "inflated":
        return np.full(max_iter, scale_factor)
    elif mode == "coarse_to_fine":
        schedule = np.ones(max_iter)
        iters_per_phase = max_iter // n_phases
        for phase in range(n_phases):
            start_it = phase * iters_per_phase
            end_it = (phase + 1) * iters_per_phase if phase < n_phases - 1 else max_iter
            phase_scale = scale_factor ** (1.0 - phase / (n_phases - 1))
            schedule[start_it:end_it] = phase_scale
        return schedule
    elif mode == "smooth_decay":
        t = np.linspace(0, 1, max_iter)
        return 1.0 + (scale_factor - 1.0) * np.exp(-5 * t)
    elif mode == "error_adaptive":
        return None
    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_convex_hull(camtoworlds):
    """Build a Delaunay triangulation from training camera positions for inside-hull testing."""
    positions = camtoworlds[:, :3, 3]  # (N, 3)
    try:
        hull = Delaunay(positions)
        return hull
    except Exception:
        # Fallback: if cameras are nearly coplanar, Delaunay fails.
        # Return None and skip the check.
        return None


def is_inside_hull(hull, point):
    """Check if a 3D point lies inside the convex hull (via Delaunay)."""
    if hull is None:
        return True  # Skip check if hull couldn't be built
    return hull.find_simplex(point) >= 0


def perturb_pose(c2w, t_range, r_range, rng):
    """Apply a random SE(3) perturbation to a camera-to-world matrix."""
    c2w_perturbed = c2w.copy()
    # Random translation
    dt = rng.uniform(-t_range, t_range, size=3)
    c2w_perturbed[:3, 3] += dt
    # Random rotation
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-8
    angle = rng.uniform(-np.radians(r_range), np.radians(r_range))
    dR = Rot.from_rotvec(axis * angle).as_matrix()
    c2w_perturbed[:3, :3] = dR @ c2w_perturbed[:3, :3]
    return c2w_perturbed


def perturb_pose_in_hull(c2w, t_range, r_range, rng, hull, max_attempts=50):
    """
    Apply a random SE(3) perturbation, rejecting samples whose position
    falls outside the convex hull of training cameras.
    """
    for _ in range(max_attempts):
        c2w_p = perturb_pose(c2w, t_range, r_range, rng)
        if is_inside_hull(hull, c2w_p[:3, 3]):
            return c2w_p
    # If all attempts fail, return the closest valid perturbation found
    # (fallback: just use a smaller perturbation)
    return perturb_pose(c2w, t_range * 0.5, r_range * 0.5, rng)


# ============================================================
# Single VS run (headless, no viewer)
# ============================================================
def run_single_vs(mode, scale_factor, cMo_start, cMo_goal, c2w_goal,
                  means, quats, scales_original, opacities, colors,
                  sh_degree, K_np, W, H, cam_params,
                  camera_model, feature_type,
                  mu, lambda_, max_iter, convergence_threshold,
                  device="cuda"):
    """
    Run a single VS experiment. Returns dict with metrics.
    """
    scale_schedule = compute_scale_schedule(mode, scale_factor, max_iter)

    wMo = np.eye(4)
    robot = SimulatorCamera()
    cMo = cMo_start.copy()
    robot.setPosition(wMo @ np.linalg.inv(cMo))
    robot.setRobotState(1)

    errors = []
    converged = False
    error_adaptive_e0 = None

    t0 = time.time()

    for it in range(max_iter):
        # Determine scale
        if scale_schedule is not None:
            sf = scale_schedule[it]
        else:
            if error_adaptive_e0 is None:
                sf = scale_factor
            else:
                ratio = min(errors[-1] / error_adaptive_e0, 1.0)
                sf = 1.0 + (scale_factor - 1.0) * ratio

        scales_current = scales_original * sf

        # Render desired at current scale
        _, gray_des, depth_des = render_gsplat(
            cMo_goal, means, quats, scales_current, opacities, colors,
            sh_degree, K_np, W, H, camera_model=camera_model, device=device,
        )
        s_star = create_feature(feature_type, device=device, border=10)
        s_star.init(H, W)
        s_star.setCameraParameters(cam_params)
        s_star.buildFrom(gray_des, depth_des)

        # Render current at current scale
        _, gray_cur, depth_cur = render_gsplat(
            cMo, means, quats, scales_current, opacities, colors,
            sh_degree, K_np, W, H, camera_model=camera_model, device=device,
        )
        s = create_feature(feature_type, device=device, border=10)
        s.init(H, W)
        s.setCameraParameters(cam_params)
        s.buildFrom(gray_cur, depth_cur)

        # Error
        error = s.error(s_star)
        err_norm = torch.sum(error ** 2).item()
        errors.append(err_norm)

        if error_adaptive_e0 is None and mode == "error_adaptive":
            error_adaptive_e0 = err_norm

        # Velocity (LM)
        Ls = s.interaction()
        Hs = Ls.T @ Ls
        diagHs = torch.diag(torch.diag(Hs))
        Hess = torch.linalg.inv(mu * diagHs + Hs + 1e-6 * torch.eye(6, device=device))
        v = -lambda_ * (Hess @ Ls.T @ error)
        v_np = v.detach().cpu().numpy()

        vt = np.linalg.norm(v_np[:3])
        vr = np.linalg.norm(v_np[3:])
        if vt > 0.5:
            v_np[:3] *= 0.5 / vt
        if vr > 0.3:
            v_np[3:] *= 0.3 / vr

        # Convergence check
        c2w_cur = np.linalg.inv(cMo)
        t_err, r_err = se3_distance(c2w_cur, c2w_goal)
        if sf < 1.1 and err_norm < convergence_threshold:
            converged = True
            break
        elif t_err < 0.005 and r_err < 0.5:
            converged = True
            break

        # Divergence check — if error grows 10x from initial, abort
        if len(errors) > 10 and err_norm > errors[0] * 10:
            break

        robot.setVelocity("camera", v_np)
        wMc = robot.getPosition()
        cMo = np.linalg.inv(wMc) @ wMo

    elapsed = time.time() - t0

    # Final pose error
    c2w_final = np.linalg.inv(cMo)
    t_err, r_err = se3_distance(c2w_final, c2w_goal)

    return {
        "converged": converged,
        "iterations": len(errors),
        "final_error": errors[-1],
        "final_t_err": float(t_err),
        "final_r_err": float(r_err),
        "time_s": elapsed,
        "errors": errors,
    }


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(
        description="Batch evaluation: Scale-Adaptive PVS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--data_factor", type=int, default=4)
    p.add_argument("--camera_model", default="pinhole")
    p.add_argument("--feature_type", default="pinhole")

    p.add_argument("--n_trials", type=int, default=20,
                   help="Number of random trials per level")
    p.add_argument("--levels", nargs="+", default=["small", "medium", "large"],
                   choices=list(PERTURBATION_LEVELS.keys()))
    p.add_argument("--modes", nargs="+", default=MODES_DEFAULT)
    p.add_argument("--scale_factor", type=float, default=1.8)

    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--lambda_", type=float, default=10.0)
    p.add_argument("--convergence_threshold", type=float, default=10000)
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out_dir", type=str, default="logs/scale_evaluation")
    args = p.parse_args()
    device = "cuda"

    rng = np.random.default_rng(args.seed)

    # ---- Load scene ----
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

    # Build convex hull of training camera positions
    hull = build_convex_hull(camtoworlds)
    if hull is not None:
        print(f"[Hull]  Convex hull built from {n_views} camera positions")
    else:
        print(f"[Hull]  WARNING: Could not build convex hull (cameras may be coplanar)")

    print(f"[Scene] {n_views} views, {W}x{H} (factor={df})")
    print(f"[Eval]  {args.n_trials} trials × {len(args.levels)} levels × {len(args.modes)} modes")
    print(f"[Modes] {args.modes}")
    print(f"[Scale] factor={args.scale_factor}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Run trials ----
    all_results = []

    for level in args.levels:
        preset = PERTURBATION_LEVELS[level]
        t_range, r_range = preset["t_range"], preset["r_range"]
        print(f"{'='*70}")
        print(f"Level: {level.upper()} (±{t_range:.2f}m, ±{r_range:.0f}°)")
        print(f"{'='*70}")

        level_results = []

        for trial in range(args.n_trials):
            # Pick a random goal view
            goal_idx = rng.integers(0, n_views)
            c2w_goal = camtoworlds[goal_idx]
            cMo_goal = np.linalg.inv(c2w_goal)

            # Perturb to get start pose (constrained to convex hull)
            c2w_start = perturb_pose_in_hull(c2w_goal, t_range, r_range, rng, hull)
            cMo_start = np.linalg.inv(c2w_start)
            init_t, init_r = se3_distance(c2w_start, c2w_goal)

            trial_data = {
                "trial": trial,
                "level": level,
                "goal_idx": int(goal_idx),
                "init_t": float(init_t),
                "init_r": float(init_r),
                "methods": {},
            }

            for mode in args.modes:
                result = run_single_vs(
                    mode=mode,
                    scale_factor=args.scale_factor,
                    cMo_start=cMo_start,
                    cMo_goal=cMo_goal,
                    c2w_goal=c2w_goal,
                    means=means, quats=quats,
                    scales_original=scales_original,
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
                # Don't save full error curves in JSON (too large)
                result_summary = {k: v for k, v in result.items() if k != "errors"}
                trial_data["methods"][mode] = result_summary

                status = "OK" if result["converged"] else "FAIL"
                print(f"  [{level:>6}] trial {trial:>3} | {mode:>16} | "
                      f"{status:>4} | it={result['iterations']:>4} | "
                      f"t={result['final_t_err']:.4f}m r={result['final_r_err']:.2f}° | "
                      f"{result['time_s']:.1f}s")

            level_results.append(trial_data)
            all_results.append(trial_data)

        # Print level summary
        print(f"\n--- {level.upper()} Summary ---")
        for mode in args.modes:
            successes = [t for t in level_results if t["methods"][mode]["converged"]]
            n_success = len(successes)
            rate = 100 * n_success / len(level_results)
            if n_success > 0:
                avg_iter = np.mean([t["methods"][mode]["iterations"] for t in successes])
                avg_t = np.mean([t["methods"][mode]["final_t_err"] for t in successes])
                avg_r = np.mean([t["methods"][mode]["final_r_err"] for t in successes])
                print(f"  {mode:>16}: {rate:5.1f}% ({n_success}/{len(level_results)}) | "
                      f"avg iter={avg_iter:.0f} | avg t={avg_t:.4f}m r={avg_r:.2f}°")
            else:
                print(f"  {mode:>16}: {rate:5.1f}% ({n_success}/{len(level_results)})")
        print()

    # ---- Save results ----
    results_path = os.path.join(args.out_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "n_trials": args.n_trials,
                "levels": args.levels,
                "modes": args.modes,
                "scale_factor": args.scale_factor,
                "max_iter": args.max_iter,
                "convergence_threshold": args.convergence_threshold,
                "seed": args.seed,
                "feature_type": args.feature_type,
                "camera_model": args.camera_model,
            },
            "trials": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ---- Final summary table ----
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY (scale_factor={args.scale_factor})")
    print(f"{'='*80}")
    header = f"{'Mode':>18}"
    for level in args.levels:
        header += f" | {level:>12}"
    print(header)
    print("-" * len(header))

    for mode in args.modes:
        row = f"{mode:>18}"
        for level in args.levels:
            trials = [t for t in all_results if t["level"] == level]
            n_success = sum(1 for t in trials if t["methods"][mode]["converged"])
            rate = 100 * n_success / len(trials) if trials else 0
            row += f" | {rate:>10.1f}%"
        print(row)
    print()


if __name__ == "__main__":
    main()
