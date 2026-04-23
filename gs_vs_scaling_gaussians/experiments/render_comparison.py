"""
Render Comparison: desired | current | diff
=============================================

For original and inflated modes, runs VS and saves a concatenated
[desired | current | diff] image to disk at each iteration (or every N iters).
Useful for visualizing how scaling Gaussians affects the convergence.

Usage:
    python -m gs_vs_scaling_gaussians.experiments.render_comparison \
        --ckpt <checkpoint> --cfg <config> \
        --start_idx 0 --goal_idx 3 --displace_z 0.15 \
        --scale_factor 1.8 --data_factor 4 \
        --save_every 50 --max_iter 500 \
        --out_dir logs/render_comparison
"""

import argparse
import os
import torch
import numpy as np
import cv2

import sys

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


def to_uint8(tensor):
    """Convert [0,1] float tensor (H,W,3) to uint8 BGR numpy."""
    img = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def diff_image(rgb_a, rgb_b):
    """Compute absolute difference, amplified and colorized for visibility."""
    diff = torch.abs(rgb_a - rgb_b)
    # Amplify for visibility (×3) and clamp
    diff_amp = torch.clamp(diff * 3.0, 0.0, 1.0)
    return diff_amp


def make_concat_image(rgb_des, rgb_cur, rgb_diff, it, err, t_err, r_err, mode, sf):
    """Create [desired | current | diff] with text overlay."""
    des_bgr = to_uint8(rgb_des)
    cur_bgr = to_uint8(rgb_cur)
    diff_bgr = to_uint8(rgb_diff)

    H, W = des_bgr.shape[:2]

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1

    cv2.putText(des_bgr, "Desired", (5, 20), font, font_scale, color, thickness)
    cv2.putText(cur_bgr, "Current", (5, 20), font, font_scale, color, thickness)
    cv2.putText(diff_bgr, "|Diff| x3", (5, 20), font, font_scale, color, thickness)

    # Add iteration/error info on current image
    info = f"it={it} err={err:.0f}"
    pose_info = f"t={t_err:.4f}m r={r_err:.1f}deg"
    if "pgm" in mode:
        mode_info = f"{mode} lambda_g={sf:.2f}"
    else:
        mode_info = f"{mode} alpha={sf:.2f}"
    cv2.putText(cur_bgr, info, (5, H - 35), font, font_scale, color, thickness)
    cv2.putText(cur_bgr, pose_info, (5, H - 20), font, font_scale, color, thickness)
    cv2.putText(cur_bgr, mode_info, (5, H - 5), font, font_scale, (0, 255, 255), thickness)

    # Concatenate horizontally
    concat = np.concatenate([des_bgr, cur_bgr, diff_bgr], axis=1)
    return concat


def compute_scale_schedule(mode, scale_factor, max_iter, n_phases=5):
    """Compute per-iteration scale multiplier."""
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
        return None  # computed online
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_mode(mode, scale_factor, c2w_start, c2w_goal, cMo_start, cMo_goal,
             means, quats, scales_original, opacities, colors,
             sh_degree, K_np, W, H, cam_params,
             camera_model, feature_type,
             mu, lambda_, max_iter, convergence_threshold,
             save_every, out_dir, device="cuda"):
    """Run VS for one mode, saving concatenated images."""

    os.makedirs(out_dir, exist_ok=True)

    scale_schedule = compute_scale_schedule(mode, scale_factor, max_iter)
    error_adaptive_e0 = None

    wMo = np.eye(4)
    robot = SimulatorCamera()
    cMo = cMo_start.copy()
    robot.setPosition(wMo @ np.linalg.inv(cMo))
    robot.setRobotState(1)

    sf_display = scale_factor if mode != "original" else 1.0
    print(f"\n[{mode}] Running (max sf={sf_display:.2f})...")
    print(f"{'Iter':>6} {'Scale':>8} {'Error':>12} {'d(m)':>8} {'r(°)':>8}")
    print("-" * 50)

    prev_err_norm = None

    for it in range(max_iter):
        # Determine current scale factor
        if scale_schedule is not None:
            sf = scale_schedule[it]
        else:
            # Error-adaptive
            if error_adaptive_e0 is None:
                sf = scale_factor
            else:
                ratio = min(prev_err_norm / error_adaptive_e0, 1.0)
                sf = 1.0 + (scale_factor - 1.0) * ratio

        scales_current = scales_original * sf

        # Render desired at current scale
        rgb_des, gray_des, depth_des = render_gsplat(
            cMo_goal, means, quats, scales_current, opacities, colors,
            sh_degree, K_np, W, H, camera_model=camera_model, device=device,
        )

        # Build desired feature
        s_star = create_feature(feature_type, device=device, border=10)
        s_star.init(H, W)
        s_star.setCameraParameters(cam_params)
        s_star.buildFrom(gray_des, depth_des)

        # Render current at current scale
        rgb_cur, gray_cur, depth_cur = render_gsplat(
            cMo, means, quats, scales_current, opacities, colors,
            sh_degree, K_np, W, H, camera_model=camera_model, device=device,
        )

        # Build current feature
        s = create_feature(feature_type, device=device, border=10)
        s.init(H, W)
        s.setCameraParameters(cam_params)
        s.buildFrom(gray_cur, depth_cur)

        # Error
        error = s.error(s_star)
        err_norm = torch.sum(error ** 2).item()
        prev_err_norm = err_norm

        if error_adaptive_e0 is None and mode == "error_adaptive":
            error_adaptive_e0 = err_norm

        # Pose error
        c2w_cur = np.linalg.inv(cMo)
        t_err, r_err = se3_distance(c2w_cur, c2w_goal)

        # Save concatenated image
        if it % save_every == 0:
            rgb_diff = diff_image(rgb_des, rgb_cur)
            concat = make_concat_image(
                rgb_des, rgb_cur, rgb_diff,
                it, err_norm, t_err, r_err, mode, sf,
            )
            path = os.path.join(out_dir, f"frame_{it:05d}.png")
            cv2.imwrite(path, concat)

            print(f"{it:>6} {sf:>8.2f} {err_norm:>12.0f} "
                  f"{t_err:>8.4f} {r_err:>8.2f}")

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

        # Convergence (photometric error only)
        if convergence_threshold > 0 and err_norm < convergence_threshold:
            print(f"  CONVERGED at it={it+1}")
            rgb_diff = diff_image(rgb_des, rgb_cur)
            concat = make_concat_image(rgb_des, rgb_cur, rgb_diff,
                                       it, err_norm, t_err, r_err, mode, sf)
            cv2.imwrite(os.path.join(out_dir, f"frame_{it:05d}.png"), concat)
            break

        robot.setVelocity("camera", v_np)
        wMc = robot.getPosition()
        cMo = np.linalg.inv(wMc) @ wMo

    n_frames = len([f for f in os.listdir(out_dir) if f.startswith("frame_")])
    print(f"  Saved {n_frames} frames to {out_dir}/")


def run_pgm_mode(c2w_start, c2w_goal, cMo_start, cMo_goal,
                 means, quats, scales_original, opacities, colors,
                 sh_degree, K_np, W, H, cam_params,
                 camera_model, lambda_levels, gain_pgm, mu_lm,
                 max_iter, convergence_threshold, save_every, out_dir, device="cuda"):
    """Run multi-level PGM-VS, saving [desired|current|diff] frames with jet colormap."""
    os.makedirs(out_dir, exist_ok=True)

    # Render desired once (PGM uses original scales)
    rgb_des_orig, gray_des, depth_des = render_gsplat(
        cMo_goal, means, quats, scales_original, opacities, colors,
        sh_degree, K_np, W, H, camera_model=camera_model, device=device)

    wMo = np.eye(4)
    robot = SimulatorCamera()
    cMo = cMo_start.copy()
    robot.setPosition(wMo @ np.linalg.inv(cMo))
    robot.setRobotState(1)

    total_it = 0
    iters_per_level = max_iter // len(lambda_levels)

    print(f"\n[pgm_vs] Running (levels={lambda_levels}, gain={gain_pgm})...")
    print(f"{'Iter':>6} {'Lambda':>8} {'Error':>12} {'d(m)':>8} {'r(°)':>8}")
    print("-" * 50)

    for lam in lambda_levels:
        s_star = FeaturePGM(lambda_g=lam, border=10, device=device)
        s_star.init(H, W)
        s_star.setCameraParameters(cam_params)
        s_star.buildFrom(gray_des, depth_des)

        prev_mse = float('inf')
        stall_count = 0

        for it in range(iters_per_level):
            # Render current (original scales — PGM does its own smoothing)
            rgb_cur, gray_cur, depth_cur = render_gsplat(
                cMo, means, quats, scales_original, opacities, colors,
                sh_degree, K_np, W, H, camera_model=camera_model, device=device)

            s = FeaturePGM(lambda_g=lam, border=10, device=device)
            s.init(H, W)
            s.setCameraParameters(cam_params)
            s.buildFrom(gray_cur, depth_cur)

            error = s.error(s_star)
            err_norm = torch.sum(error ** 2).item()
            mse = err_norm / max(error.shape[0], 1)

            c2w_cur = np.linalg.inv(cMo)
            t_err, r_err = se3_distance(c2w_cur, c2w_goal)

            # Save frame showing PGM features with jet colormap
            def save_pgm_frame(it_num):
                border_p = 10
                h_c, w_c = H - 2 * border_p, W - 2 * border_p
                pgm_des_2d = s_star.G.reshape(h_c, w_c).cpu().numpy()
                pgm_cur_2d = s.G.reshape(h_c, w_c).cpu().numpy()
                pgm_max = max(pgm_des_2d.max(), pgm_cur_2d.max(), 1e-8)
                # Apply jet colormap (like Guerbas et al. Fig. 1)
                des_jet = cv2.applyColorMap(
                    (pgm_des_2d / pgm_max * 255).clip(0, 255).astype(np.uint8),
                    cv2.COLORMAP_JET)
                cur_jet = cv2.applyColorMap(
                    (pgm_cur_2d / pgm_max * 255).clip(0, 255).astype(np.uint8),
                    cv2.COLORMAP_JET)
                pgm_diff = np.abs(pgm_cur_2d - pgm_des_2d)
                diff_jet = cv2.applyColorMap(
                    (pgm_diff / (pgm_max * 0.3 + 1e-8) * 255).clip(0, 255).astype(np.uint8),
                    cv2.COLORMAP_HOT)
                # Add text
                font = cv2.FONT_HERSHEY_SIMPLEX
                Hh = des_jet.shape[0]
                cv2.putText(des_jet, "Desired PGM", (5, 20), font, 0.5, (255,255,255), 1)
                cv2.putText(cur_jet, "Current PGM", (5, 20), font, 0.5, (255,255,255), 1)
                cv2.putText(diff_jet, "|Diff| PGM", (5, 20), font, 0.5, (255,255,255), 1)
                cv2.putText(cur_jet, f"it={it_num} err={err_norm:.0f}", (5, Hh-35), font, 0.5, (255,255,255), 1)
                cv2.putText(cur_jet, f"t={t_err:.4f}m r={r_err:.1f}deg", (5, Hh-20), font, 0.5, (255,255,255), 1)
                cv2.putText(cur_jet, f"pgm lambda_g={lam:.1f}", (5, Hh-5), font, 0.5, (0,255,255), 1)
                concat = np.concatenate([des_jet, cur_jet, diff_jet], axis=1)
                cv2.imwrite(os.path.join(out_dir, f"frame_{it_num:05d}.png"), concat)

            if total_it % save_every == 0:
                save_pgm_frame(total_it)
                print(f"{total_it:>6} {lam:>8.1f} {err_norm:>12.0f} "
                      f"{t_err:>8.4f} {r_err:>8.2f}")

            total_it += 1

            # Convergence (photometric error only)
            if convergence_threshold > 0 and err_norm < convergence_threshold:
                print(f"  CONVERGED at it={total_it}")
                save_pgm_frame(total_it)
                n_frames = len([f for f in os.listdir(out_dir) if f.startswith("frame_")])
                print(f"  Saved {n_frames} frames to {out_dir}/")
                return

            # Stall detection
            if mse >= prev_mse * 0.999:
                stall_count += 1
            else:
                stall_count = 0
            prev_mse = mse
            if stall_count > 50:
                break

            # LM control
            LG = s.interaction()
            LtL = LG.T @ LG
            H_inv = torch.linalg.inv(
                mu_lm * torch.diag(torch.diag(LtL)) + LtL
                + 1e-6 * torch.eye(6, device=device))
            v = -gain_pgm * (H_inv @ LG.T @ error)
            v_np = v.detach().cpu().numpy()

            vt, vr = np.linalg.norm(v_np[:3]), np.linalg.norm(v_np[3:])
            if vt > 0.5: v_np[:3] *= 0.5 / vt
            if vr > 0.3: v_np[3:] *= 0.3 / vr

            robot.setVelocity("camera", v_np)
            wMc = robot.getPosition()
            cMo = np.linalg.inv(wMc) @ wMo

    n_frames = len([f for f in os.listdir(out_dir) if f.startswith("frame_")])
    print(f"  FAILED after {total_it} iterations")
    print(f"  Saved {n_frames} frames to {out_dir}/")


def main():
    p = argparse.ArgumentParser(
        description="Render comparison: desired | current | diff",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--start_idx", type=int, required=True)
    p.add_argument("--goal_idx", type=int, required=True)
    p.add_argument("--data_factor", type=int, default=4)
    p.add_argument("--camera_model", default="pinhole")
    p.add_argument("--feature_type", default="pinhole")

    p.add_argument("--scale_factor", type=float, default=1.8)
    p.add_argument("--displace_z", type=float, default=0.0)

    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--lambda_", type=float, default=10.0)
    p.add_argument("--convergence_threshold", type=float, default=100,
                   help="Photometric error threshold for original/c2f modes")
    p.add_argument("--convergence_threshold_inflated", type=float, default=0.1,
                   help="Photometric error threshold for inflated mode (0=run to max_iter)")
    p.add_argument("--convergence_threshold_pgm", type=float, default=100,
                   help="Photometric error threshold for PGM-VS mode")
    p.add_argument("--max_iter", type=int, default=500)
    p.add_argument("--save_every", type=int, default=10,
                   help="Save a frame every N iterations")

    p.add_argument("--out_dir", type=str, default="logs/render_comparison")
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

    pixel_ratio = (W * H) / (W_full * H_full)
    thresh_original = args.convergence_threshold * pixel_ratio
    thresh_inflated = args.convergence_threshold_inflated * pixel_ratio
    thresh_pgm = args.convergence_threshold_pgm * pixel_ratio

    c2w_start = camtoworlds[args.start_idx].copy()
    c2w_goal = camtoworlds[args.goal_idx]

    if args.displace_z != 0.0:
        z_axis = c2w_start[:3, 2]
        c2w_start[:3, 3] -= z_axis * args.displace_z
        print(f"[Displace] Start moved {args.displace_z:.3f}m along optical axis")

    t_dist, r_dist = se3_distance(c2w_start, c2w_goal)
    cMo_start = np.linalg.inv(c2w_start)
    cMo_goal = np.linalg.inv(c2w_goal)

    print(f"[Scene] {len(camtoworlds)} views, {W}x{H} (factor={df})")
    print(f"[Pair]  {args.start_idx} → {args.goal_idx}: d={t_dist:.3f}m, r={r_dist:.1f}°")
    print(f"[Save]  every {args.save_every} iterations")

    # Run scaling modes with per-mode thresholds
    mode_thresholds = {"original": thresh_original, "inflated": thresh_inflated}
    for mode in ["original", "inflated"]:
        run_mode(
            mode=mode,
            scale_factor=args.scale_factor,
            c2w_start=c2w_start, c2w_goal=c2w_goal,
            cMo_start=cMo_start, cMo_goal=cMo_goal,
            means=means, quats=quats, scales_original=scales_original,
            opacities=opacities, colors=colors,
            sh_degree=sh_degree, K_np=K_np, W=W, H=H,
            cam_params=cam_params,
            camera_model=args.camera_model,
            feature_type=args.feature_type,
            mu=args.mu, lambda_=args.lambda_,
            max_iter=args.max_iter,
            convergence_threshold=mode_thresholds[mode],
            save_every=args.save_every,
            out_dir=os.path.join(args.out_dir, mode),
            device=device,
        )

    # Run PGM-VS
    run_pgm_mode(
        c2w_start=c2w_start, c2w_goal=c2w_goal,
        cMo_start=cMo_start, cMo_goal=cMo_goal,
        means=means, quats=quats, scales_original=scales_original,
        opacities=opacities, colors=colors,
        sh_degree=sh_degree, K_np=K_np, W=W, H=H,
        cam_params=cam_params,
        camera_model=args.camera_model,
        lambda_levels=[5.0, 2.5, 1.0],
        gain_pgm=10.0, mu_lm=0.01,
        max_iter=args.max_iter,
        convergence_threshold=thresh_pgm,
        save_every=args.save_every,
        out_dir=os.path.join(args.out_dir, "pgm_vs"),
        device=device,
    )

    all_modes = modes + ["pgm_vs"]
    print(f"\nDone. Results in {args.out_dir}/")
    for m in all_modes:
        print(f"  {args.out_dir}/{m}/")


if __name__ == "__main__":
    main()
