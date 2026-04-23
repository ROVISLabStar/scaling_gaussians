"""
PGM-VS vs Scale-Adaptive Comparison
=====================================

Head-to-head comparison on the same view pairs.
For each pair, runs:
  - Original PVS (baseline)
  - Inflated PVS (scale_factor × Gaussians)
  - PGM-VS with adaptive lambda_g (Crombez TRO 2019)

Records iterations, final pose error, convergence status.

Usage:
    python -m gs_vs_scaling_gaussians.experiments.pgm_comparison \
        --ckpt <checkpoint> --cfg <config> \
        --pairs "0:4,40:44,10:14,65:68,130:133,150:153" \
        --scale_factor 1.8 --lambda_gi 25.0 \
        --out_dir logs/pgm_comparison
"""

import argparse
import json
import os
import sys
import time
import torch
import numpy as np

from gs_vs.simulators.RobotSimulator import SimulatorCamera
from gs_vs.simulators.CameraParameters import CameraParameters
from gs_vs.features.factory import create_feature
from gs_vs.datasets.colmap import Parser
from gsplat.rendering import rasterization

# Import PGM feature
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'gs_vs_pgm_vs'))
from features.FeaturePGM import FeaturePGM


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


# ============================================================
# VS runners
# ============================================================

def run_pvs(mode, scale_factor, cMo_start, cMo_goal, c2w_goal,
            means, quats, scales_original, opacities, colors,
            sh_degree, K_np, W, H, cam_params,
            camera_model, feature_type, mu, lambda_gain,
            max_iter, device="cuda"):
    """Run standard PVS (original or inflated)."""
    sf = 1.0 if mode == "original" else scale_factor

    wMo = np.eye(4)
    robot = SimulatorCamera()
    cMo = cMo_start.copy()
    robot.setPosition(wMo @ np.linalg.inv(cMo))
    robot.setRobotState(1)

    scales_current = scales_original * sf

    for it in range(max_iter):
        # Render desired
        _, gray_des, depth_des = render_gsplat(
            cMo_goal, means, quats, scales_current, opacities, colors,
            sh_degree, K_np, W, H, camera_model=camera_model, device=device)
        s_star = create_feature(feature_type, device=device, border=10)
        s_star.init(H, W)
        s_star.setCameraParameters(cam_params)
        s_star.buildFrom(gray_des, depth_des)

        # Render current
        _, gray_cur, depth_cur = render_gsplat(
            cMo, means, quats, scales_current, opacities, colors,
            sh_degree, K_np, W, H, camera_model=camera_model, device=device)
        s = create_feature(feature_type, device=device, border=10)
        s.init(H, W)
        s.setCameraParameters(cam_params)
        s.buildFrom(gray_cur, depth_cur)

        error = s.error(s_star)
        err_norm = torch.sum(error ** 2).item()

        # Pose error
        c2w_cur = np.linalg.inv(cMo)
        t_err, r_err = se3_distance(c2w_cur, c2w_goal)

        # Convergence
        if t_err < 0.005 and r_err < 0.5:
            return {"converged": True, "iterations": it + 1,
                    "final_t": float(t_err), "final_r": float(r_err)}

        # Velocity
        Ls = s.interaction()
        Hs = Ls.T @ Ls
        diagHs = torch.diag(torch.diag(Hs))
        Hess = torch.linalg.inv(mu * diagHs + Hs + 1e-6 * torch.eye(6, device=device))
        v = -lambda_gain * (Hess @ Ls.T @ error)
        v_np = v.detach().cpu().numpy()

        vt, vr = np.linalg.norm(v_np[:3]), np.linalg.norm(v_np[3:])
        if vt > 0.5: v_np[:3] *= 0.5 / vt
        if vr > 0.3: v_np[3:] *= 0.3 / vr

        robot.setVelocity("camera", v_np)
        wMc = robot.getPosition()
        cMo = np.linalg.inv(wMc) @ wMo

    c2w_final = np.linalg.inv(cMo)
    t_err, r_err = se3_distance(c2w_final, c2w_goal)
    return {"converged": False, "iterations": max_iter,
            "final_t": float(t_err), "final_r": float(r_err)}


def run_pgm_vs(cMo_start, cMo_goal, c2w_goal,
               means, quats, scales_original, opacities, colors,
               sh_degree, K_np, W, H, cam_params,
               camera_model, lambda_gi, lambda_g_final,
               mu_pgm, max_iter, switch_threshold=0.1,
               device="cuda"):
    """
    Run PGM-VS with adaptive lambda_g (Crombez TRO 2019).

    2-step strategy:
      Step 1: lambda_g_cur starts at lambda_gi, desired at lambda_gi/2.
              Both adapt via the extended interaction matrix.
              When lambda_g_cur converges to lambda_g_star → switch to step 2.
      Step 2: lambda_g = lambda_g_final (small, for precision).
    """
    wMo = np.eye(4)
    robot = SimulatorCamera()
    cMo = cMo_start.copy()
    robot.setPosition(wMo @ np.linalg.inv(cMo))
    robot.setRobotState(1)

    lambda_g_cur = lambda_gi
    lambda_g_star = lambda_gi / 2.0  # desired lambda (Step 1)
    pgm_step_num = 1

    # Pre-render desired image (constant)
    _, gray_des, depth_des = render_gsplat(
        cMo_goal, means, quats, scales_original, opacities, colors,
        sh_degree, K_np, W, H, camera_model=camera_model, device=device)

    # Build initial desired PGM feature
    s_star_pgm = FeaturePGM(lambda_g=lambda_g_star, border=10, device=device)
    s_star_pgm.init(H, W)
    s_star_pgm.setCameraParameters(cam_params)
    s_star_pgm.buildFrom(gray_des, depth_des)

    for it in range(max_iter):
        # Render current
        _, gray_cur, depth_cur = render_gsplat(
            cMo, means, quats, scales_original, opacities, colors,
            sh_degree, K_np, W, H, camera_model=camera_model, device=device)

        # Build current PGM feature
        s_pgm = FeaturePGM(lambda_g=lambda_g_cur, border=10, device=device)
        s_pgm.init(H, W)
        s_pgm.setCameraParameters(cam_params)
        s_pgm.buildFrom(gray_cur, depth_cur)

        # Error
        error = s_pgm.error(s_star_pgm)
        err_norm = torch.sum(error ** 2).item()
        n_pixels = error.shape[0]
        err_mse = err_norm / max(n_pixels, 1)

        # Pose error
        c2w_cur = np.linalg.inv(cMo)
        t_err, r_err = se3_distance(c2w_cur, c2w_goal)

        # Convergence: pose-based only (MSE threshold is unreliable
        # for PGM since blurred images have lower absolute error)
        if t_err < 0.005 and r_err < 0.5:
            return {"converged": True, "iterations": it + 1,
                    "final_t": float(t_err), "final_r": float(r_err),
                    "final_lambda": float(lambda_g_cur), "step": pgm_step_num}

        # Extended interaction (6 DoF + lambda)
        LG_ext = s_pgm.interaction_extended()  # (N, 7)
        LtL = LG_ext.T @ LG_ext
        damping = 1e-6 * torch.eye(7, device=device)
        v_lambda = -mu_pgm * torch.linalg.solve(LtL + damping, LG_ext.T @ error)

        v_np = v_lambda[:6].detach().cpu().numpy()
        delta_lambda = v_lambda[6].item()

        # NO velocity clamping for PGM (the gain handles it)

        # Update lambda
        lambda_g_cur = max(lambda_g_cur + delta_lambda, 0.1)

        # Step 1 -> Step 2 transition
        if pgm_step_num == 1 and abs(lambda_g_cur - lambda_g_star) < switch_threshold:
            pgm_step_num = 2
            lambda_g_cur = lambda_g_final
            lambda_g_star = lambda_g_final
            # Rebuild desired feature with new lambda
            s_star_pgm = FeaturePGM(lambda_g=lambda_g_star, border=10, device=device)
            s_star_pgm.init(H, W)
            s_star_pgm.setCameraParameters(cam_params)
            s_star_pgm.buildFrom(gray_des, depth_des)

        robot.setVelocity("camera", v_np)
        wMc = robot.getPosition()
        cMo = np.linalg.inv(wMc) @ wMo

    c2w_final = np.linalg.inv(cMo)
    t_err, r_err = se3_distance(c2w_final, c2w_goal)
    return {"converged": False, "iterations": max_iter,
            "final_t": float(t_err), "final_r": float(r_err),
            "final_lambda": float(lambda_g_cur), "step": pgm_step_num}


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(
        description="PGM-VS vs Scale-Adaptive comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--data_factor", type=int, default=4)
    p.add_argument("--camera_model", default="pinhole")
    p.add_argument("--feature_type", default="pinhole")

    p.add_argument("--pairs", type=str,
                   default="0:4,40:44,10:14,65:68,130:133,150:153",
                   help="Comma-separated start:goal pairs")
    p.add_argument("--scale_factor", type=float, default=1.8)
    p.add_argument("--lambda_gi", type=float, default=25.0,
                   help="PGM initial lambda_g")
    p.add_argument("--lambda_g_final", type=float, default=1.0,
                   help="PGM final lambda_g (Step 2, high accuracy)")
    p.add_argument("--mu_pgm", type=float, default=1.0,
                   help="PGM control gain")

    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--lambda_", type=float, default=10.0)
    p.add_argument("--max_iter", type=int, default=1000)

    p.add_argument("--out_dir", type=str, default="logs/pgm_comparison")
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
    W_full, H_full = list(parser.imsize_dict.values())[0]
    K_colmap = list(parser.Ks_dict.values())[0]

    df = args.data_factor
    W, H = W_full // df, H_full // df
    fx, fy = K_colmap[0, 0] / df, K_colmap[1, 1] / df
    cx, cy = K_colmap[0, 2] / df, K_colmap[1, 2] / df
    K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    cam_params = CameraParameters(px=fx, py=fy, u0=cx, v0=cy)

    # Parse pairs
    pairs = []
    for pair_str in args.pairs.split(","):
        s, g = pair_str.strip().split(":")
        pairs.append((int(s), int(g)))

    methods = ["original", "inflated", "pgm_vs"]
    print(f"[Scene] {len(camtoworlds)} views, {W}x{H}")
    print(f"[Pairs] {len(pairs)} pairs: {pairs}")
    print(f"[Methods] {methods}")
    print(f"[Params] scale_factor={args.scale_factor}, lambda_gi={args.lambda_gi}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = []

    print(f"{'Pair':>12} | {'Method':>12} | {'Result':>6} | {'Iter':>6} | {'t(m)':>8} | {'r(°)':>8}")
    print("-" * 70)

    for start_idx, goal_idx in pairs:
        c2w_goal = camtoworlds[goal_idx]
        c2w_start = camtoworlds[start_idx]
        cMo_start = np.linalg.inv(c2w_start)
        cMo_goal = np.linalg.inv(c2w_goal)
        init_t, init_r = se3_distance(c2w_start, c2w_goal)

        trial = {
            "start": start_idx, "goal": goal_idx,
            "init_t": float(init_t), "init_r": float(init_r),
            "methods": {},
        }

        # Original PVS
        t0 = time.time()
        res = run_pvs("original", 1.0, cMo_start, cMo_goal, c2w_goal,
                       means, quats, scales_original, opacities, colors,
                       sh_degree, K_np, W, H, cam_params,
                       args.camera_model, args.feature_type,
                       args.mu, args.lambda_, args.max_iter, device)
        res["time_s"] = time.time() - t0
        trial["methods"]["original"] = res
        status = "OK" if res["converged"] else "FAIL"
        print(f"{start_idx:>5}->{goal_idx:<5} | {'original':>12} | {status:>6} | "
              f"{res['iterations']:>6} | {res['final_t']:>8.4f} | {res['final_r']:>8.2f}")

        # Inflated PVS
        t0 = time.time()
        res = run_pvs("inflated", args.scale_factor, cMo_start, cMo_goal, c2w_goal,
                       means, quats, scales_original, opacities, colors,
                       sh_degree, K_np, W, H, cam_params,
                       args.camera_model, args.feature_type,
                       args.mu, args.lambda_, args.max_iter, device)
        res["time_s"] = time.time() - t0
        trial["methods"]["inflated"] = res
        status = "OK" if res["converged"] else "FAIL"
        print(f"{'':>12} | {'inflated':>12} | {status:>6} | "
              f"{res['iterations']:>6} | {res['final_t']:>8.4f} | {res['final_r']:>8.2f}")

        # PGM-VS
        t0 = time.time()
        res = run_pgm_vs(cMo_start, cMo_goal, c2w_goal,
                          means, quats, scales_original, opacities, colors,
                          sh_degree, K_np, W, H, cam_params,
                          args.camera_model, args.lambda_gi, args.lambda_g_final,
                          args.mu_pgm, args.max_iter, device=device)
        res["time_s"] = time.time() - t0
        trial["methods"]["pgm_vs"] = res
        status = "OK" if res["converged"] else "FAIL"
        print(f"{'':>12} | {'pgm_vs':>12} | {status:>6} | "
              f"{res['iterations']:>6} | {res['final_t']:>8.4f} | {res['final_r']:>8.2f}")

        print("-" * 70)
        all_results.append(trial)

    # Save
    out_path = os.path.join(args.out_dir, "pgm_comparison.json")
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "trials": all_results}, f, indent=2)

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Pair':>12} | {'Dist':>14} | {'Original':>10} | {'Inflated':>10} | {'PGM-VS':>10}")
    print("-" * 65)
    for t in all_results:
        pair = f"{t['start']}->{t['goal']}"
        dist = f"{t['init_t']:.2f}m,{t['init_r']:.0f}\u00b0"
        cols = []
        for m in ["original", "inflated", "pgm_vs"]:
            r = t["methods"][m]
            if r["converged"]:
                cols.append(f"{r['iterations']:>4} it")
            else:
                cols.append(f"{'FAIL':>6}")
        print(f"{pair:>12} | {dist:>14} | {cols[0]:>10} | {cols[1]:>10} | {cols[2]:>10}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
