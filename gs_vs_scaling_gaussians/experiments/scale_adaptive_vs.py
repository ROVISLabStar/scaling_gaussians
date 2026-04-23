"""
Scale-Adaptive Photometric Visual Servoing in 3DGS
====================================================

Demonstrates that controlling 3DGS Gaussian scales improves PVS convergence,
analogous to Crombez et al.'s (IROS 2015) λ_g parameter in PGM-VS.

Three modes:
  1. ORIGINAL: VS with trained scales (small basin, precise)
  2. INFLATED: VS with scales × factor (large basin, coarse)
  3. COARSE-TO-FINE: Start inflated, gradually reduce to original
     (large basin for approach, precise for final convergence)

Usage:
    # Original scales (will fail for large displacement)
    python experiments/scale_adaptive_vs.py \
        --ckpt ... --cfg ... --start_idx 255 --goal_idx 72 \
        --mode original --port 8080

    # Inflated scales (should converge from further)
    python experiments/scale_adaptive_vs.py \
        --ckpt ... --cfg ... --start_idx 255 --goal_idx 72 \
        --mode inflated --scale_factor 3.0 --port 8081

    # Coarse-to-fine (best of both worlds)
    python experiments/scale_adaptive_vs.py \
        --ckpt ... --cfg ... --start_idx 255 --goal_idx 72 \
        --mode coarse_to_fine --port 8082

Author: Youssef (UM6P / Ai Movement Lab)
"""

import argparse
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


def save_rgb(tensor, path):
    import cv2
    img = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


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
    """
    Compute scale multiplier at each iteration.

    Returns:
        schedule: array of length max_iter, scale multiplier at each step
    """
    if mode == "original":
        return np.ones(max_iter)

    elif mode == "inflated":
        return np.full(max_iter, scale_factor)

    elif mode == "coarse_to_fine":
        # Exponential decay from scale_factor to 1.0
        # Split into phases, each phase halves the excess
        schedule = np.ones(max_iter)
        iters_per_phase = max_iter // n_phases

        for phase in range(n_phases):
            start_it = phase * iters_per_phase
            end_it = (phase + 1) * iters_per_phase if phase < n_phases - 1 else max_iter

            # Scale factor decays: sf, sf^(1/2), sf^(1/4), ..., 1.0
            phase_scale = scale_factor ** (1.0 - phase / (n_phases - 1))
            schedule[start_it:end_it] = phase_scale

        return schedule

    elif mode == "smooth_decay":
        # Smooth exponential decay from scale_factor to 1.0
        t = np.linspace(0, 1, max_iter)
        schedule = 1.0 + (scale_factor - 1.0) * np.exp(-5 * t)
        return schedule

    elif mode == "error_adaptive":
        # Placeholder — actual scale is computed online from error magnitude.
        # Return None to signal the loop to use adaptive logic.
        return None

    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    p = argparse.ArgumentParser(
        description="Scale-Adaptive PVS in 3DGS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--start_idx", type=int, required=True)
    p.add_argument("--goal_idx", type=int, required=True)
    p.add_argument("--data_factor", type=int, default=4)
    p.add_argument("--camera_model", default="pinhole")
    p.add_argument("--feature_type", default="pinhole")

    # Scale control
    p.add_argument("--mode", default="original",
                   choices=["original", "inflated", "coarse_to_fine", "smooth_decay",
                            "error_adaptive"],
                   help="Scale adaptation mode")
    p.add_argument("--scale_factor", type=float, default=3.0,
                   help="Scale multiplier (for inflated/coarse_to_fine modes)")

    # VS parameters
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--lambda_", type=float, default=10.0)
    p.add_argument("--convergence_threshold", type=float, default=10000)
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--displace_z", type=float, default=0.0,
                   help="Displace start pose along optical axis (meters). "
                        "Positive = move backward, negative = move forward. "
                        "If set, start_idx is used as base pose and displaced.")

    # Output
    p.add_argument("--out_dir", type=str, default="logs/scale_vs")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--headless", action="store_true",
                   help="Run without viser viewer (for batch/scripted runs)")
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
    fx_full, fy_full = K_colmap[0, 0], K_colmap[1, 1]
    cx_full, cy_full = K_colmap[0, 2], K_colmap[1, 2]

    # Apply data_factor
    df = args.data_factor
    W = W_full // df
    H = H_full // df
    fx, fy = fx_full / df, fy_full / df
    cx, cy = cx_full / df, cy_full / df
    K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    cam_params = CameraParameters(px=fx, py=fy, u0=cx, v0=cy)

    # Scale threshold
    pixel_ratio = (W * H) / (W_full * H_full)
    convergence_threshold = args.convergence_threshold * pixel_ratio

    c2w_start = camtoworlds[args.start_idx]
    c2w_goal = camtoworlds[args.goal_idx]

    # Apply Z displacement to start pose
    if args.displace_z != 0.0:
        z_axis = c2w_start[:3, 2]  # optical axis (OpenCV: camera looks along -Z)
        c2w_start = c2w_start.copy()
        c2w_start[:3, 3] -= z_axis * args.displace_z  # positive = move backward
        print(f"[Displace] Start moved {args.displace_z:.3f}m along optical axis")

    t_dist, r_dist = se3_distance(c2w_start, c2w_goal)

    print(f"[Scene] {len(camtoworlds)} views, {W}x{H} (factor={df})")
    print(f"[Pair] view {args.start_idx} → {args.goal_idx}: d={t_dist:.3f}m, r={r_dist:.1f}°")
    print(f"[Mode] {args.mode} (scale_factor={args.scale_factor})")
    print(f"[VS] max_iter={args.max_iter}, threshold={convergence_threshold:.0f}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Compute scale schedule
    scale_schedule = compute_scale_schedule(
        args.mode, args.scale_factor, args.max_iter
    )
    if scale_schedule is not None:
        print(f"[Schedule] Scale range: {scale_schedule[0]:.2f} → {scale_schedule[-1]:.2f}")
    else:
        print(f"[Schedule] Error-adaptive (max scale={args.scale_factor:.2f})")
    # For error-adaptive: initial error will calibrate the mapping
    error_adaptive_e0 = None

    # Render desired image at ORIGINAL scales (target is always sharp)
    cMo_goal = np.linalg.inv(c2w_goal)
    rgb_goal, gray_goal, depth_goal = render_gsplat(
        cMo_goal, means, quats, scales_original, opacities, colors,
        sh_degree, K_np, W, H, camera_model=args.camera_model,
    )
    save_rgb(rgb_goal, os.path.join(args.out_dir, "goal.png"))

    # Render start
    cMo_start = np.linalg.inv(c2w_start)
    rgb_start, _, _ = render_gsplat(
        cMo_start, means, quats, scales_original, opacities, colors,
        sh_degree, K_np, W, H, camera_model=args.camera_model,
    )
    save_rgb(rgb_start, os.path.join(args.out_dir, "start.png"))

    # Setup viser viewer
    viewer = None
    if not args.headless:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '..', '..', 'gs_vs_planner', 'experiments'))
        from viewer_path_planner import PathPlannerViewer

        viewer = PathPlannerViewer(
            rgb_start=rgb_start.cpu().numpy(),
            rgb_goal=rgb_goal.cpu().numpy(),
            c2w_start=c2w_start,
            c2w_goal=c2w_goal,
            waypoints_c2w=None,
            training_c2w=camtoworlds,
            image_size=(W, H),
            aspect_ratio=W / H,
            server_port=args.port,
            downsample_factor=max(1, 2 // df),
        )
        print(f"\n[Viser] http://localhost:{args.port}")

    # VS loop
    wMo = np.eye(4)
    robot = SimulatorCamera()
    cMo = cMo_start.copy()
    robot.setPosition(wMo @ np.linalg.inv(cMo))
    robot.setRobotState(1)

    errors = []
    scale_history = []
    pose_errors_t = []
    pose_errors_r = []
    converged = False

    print(f"\n[VS] Running {args.mode} mode...")
    print(f"{'Iter':>6} {'Scale':>8} {'Error':>12} {'d(m)':>8} {'r(°)':>8} {'|vt|':>8} {'|vr|':>8}")
    print("-" * 65)

    for it in range(args.max_iter):
        # Determine current scale factor
        if scale_schedule is not None:
            current_scale_factor = scale_schedule[it]
        else:
            # Error-adaptive: scale ∝ normalized error
            # First iteration: use max scale, calibrate e0
            if error_adaptive_e0 is None:
                current_scale_factor = args.scale_factor
            else:
                # Map error ratio to scale: high error → high scale, low error → 1.0
                ratio = min(err_norm / error_adaptive_e0, 1.0)
                current_scale_factor = 1.0 + (args.scale_factor - 1.0) * ratio

        scales_current = scales_original * current_scale_factor

        # Render desired with CURRENT scales (both images at same scale for fair comparison)
        _, gray_des, depth_des = render_gsplat(
            cMo_goal, means, quats, scales_current, opacities, colors,
            sh_degree, K_np, W, H, camera_model=args.camera_model,
        )

        # Build desired feature at current scale
        s_star = create_feature(args.feature_type, device=device, border=10)
        s_star.init(H, W)
        s_star.setCameraParameters(cam_params)
        s_star.buildFrom(gray_des, depth_des)

        # Render current with CURRENT scales
        rgb_cur, gray_cur, depth_cur = render_gsplat(
            cMo, means, quats, scales_current, opacities, colors,
            sh_degree, K_np, W, H, camera_model=args.camera_model,
        )

        # Build current feature
        s = create_feature(args.feature_type, device=device, border=10)
        s.init(H, W)
        s.setCameraParameters(cam_params)
        s.buildFrom(gray_cur, depth_cur)

        # Compute error
        error = s.error(s_star)
        err_norm = torch.sum(error ** 2).item()
        errors.append(err_norm)
        scale_history.append(current_scale_factor)

        # Calibrate error-adaptive reference on first iteration
        if error_adaptive_e0 is None and args.mode == "error_adaptive":
            error_adaptive_e0 = err_norm

        # Pose error
        c2w_cur = np.linalg.inv(cMo)
        t_err, r_err = se3_distance(c2w_cur, c2w_goal)
        pose_errors_t.append(t_err)
        pose_errors_r.append(r_err)

        # Compute velocity
        Ls = s.interaction()
        Hs = Ls.T @ Ls
        diagHs = torch.diag(torch.diag(Hs))
        Hess = torch.linalg.inv(args.mu * diagHs + Hs + 1e-6 * torch.eye(6, device=device))
        v = -args.lambda_ * (Hess @ Ls.T @ error)
        v_np = v.detach().cpu().numpy()

        vt = np.linalg.norm(v_np[:3])
        vr = np.linalg.norm(v_np[3:])
        if vt > 0.5:
            v_np[:3] *= 0.5 / vt
        if vr > 0.3:
            v_np[3:] *= 0.3 / vr

        # Viewer update
        if viewer is not None and it % 5 == 0:
            viewer.update(
                iteration=it,
                cMo=np.linalg.inv(cMo),
                rgb=rgb_cur,
                error=err_norm,
                velocity=v_np,
            )

        # Print progress
        if it % 100 == 0:
            print(f"{it:>6} {current_scale_factor:>8.2f} {err_norm:>12.0f} "
                  f"{t_err:>8.4f} {r_err:>8.2f} {vt:>8.4f} {vr:>8.4f}")

        # Convergence check:
        # - For modes that decay to original scale: check photometric error at scale~1.0
        # - For inflated mode (constant scale): check pose error directly
        #   (photometric error at inflated scale isn't comparable to threshold)
        if current_scale_factor < 1.1:
            if err_norm < convergence_threshold:
                converged = True
                print(f"\n  ✓ CONVERGED at it={it+1} (scale={current_scale_factor:.2f})")
                print(f"    Error: {err_norm:.0f}, Pose: {t_err:.4f}m, {r_err:.2f}°")
                break
        elif t_err < 0.005 and r_err < 0.5:
            # Pose-based convergence for inflated/error_adaptive modes
            converged = True
            print(f"\n  ✓ CONVERGED (pose) at it={it+1} (scale={current_scale_factor:.2f})")
            print(f"    Error: {err_norm:.0f}, Pose: {t_err:.4f}m, {r_err:.2f}°")
            break

        robot.setVelocity("camera", v_np)
        wMc = robot.getPosition()
        cMo = np.linalg.inv(wMc) @ wMo

    if not converged:
        print(f"\n  ✗ FAILED after {args.max_iter} iterations")
        print(f"    Error: {err_norm:.0f}, Pose: {t_err:.4f}m, {r_err:.2f}°")

    # Save final image
    rgb_final, _, _ = render_gsplat(
        cMo, means, quats, scales_original, opacities, colors,
        sh_degree, K_np, W, H, camera_model=args.camera_model,
    )
    save_rgb(rgb_final, os.path.join(args.out_dir, f"final_{args.mode}.png"))

    # Save convergence data
    np.savez(
        os.path.join(args.out_dir, f"convergence_{args.mode}.npz"),
        errors=np.array(errors),
        scale_history=np.array(scale_history),
        pose_errors_t=np.array(pose_errors_t),
        pose_errors_r=np.array(pose_errors_r),
        converged=converged,
        mode=args.mode,
        scale_factor=args.scale_factor,
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULT: {args.mode} (scale_factor={args.scale_factor})")
    print(f"  Pair: {args.start_idx} → {args.goal_idx} (d={t_dist:.3f}m, r={r_dist:.1f}°)")
    print(f"  Converged: {'YES' if converged else 'NO'}")
    print(f"  Iterations: {len(errors)}")
    print(f"  Final pose error: {t_err:.4f}m, {r_err:.2f}°")
    print(f"  Images saved to {args.out_dir}/")
    print(f"{'='*60}")

    if not args.headless:
        print(f"\n[Viser] Viewer running at http://localhost:{args.port}")
        print("  Press Ctrl+C to exit")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nDone.")


if __name__ == "__main__":
    main()
