"""
DDVS Visual Servoing with 3DGS rendering
==========================================

Simulates defocus-based DVS using depth maps from gsplat rendering.
The thin lens model creates depth-dependent blur controlled by
aperture (D) and focus depth (Z_f).

Compares: Original PVS vs DDVS vs Inflated 3DGS.

Usage:
    python -m gs_vs_ddvs.experiments.ddvs_servo \
        --ckpt <checkpoint> --cfg <config> \
        --start_idx 14 --goal_idx 10 \
        --aperture_phi 2.0 --focus_depth 0.5 \
        --out_dir logs/ddvs
"""

import argparse
import os
import sys
import torch
import numpy as np
import cv2

from gs_vs.simulators.RobotSimulator import SimulatorCamera
from gs_vs.simulators.CameraParameters import CameraParameters
from gs_vs.features.factory import create_feature
from gs_vs.datasets.colmap import Parser
from gsplat.rendering import rasterization

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.FeatureDDVS import FeatureDDVS

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(PROJ_DIR)))
from gs_vs_pgm_vs.features.FeaturePGM import FeaturePGM


def se3_distance(c2w_a, c2w_b):
    t_dist = np.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3])
    R_rel = c2w_a[:3, :3].T @ c2w_b[:3, :3]
    angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
    return t_dist, np.degrees(angle)


@torch.no_grad()
def render_gsplat(cMo, means, quats, scales, opacities, colors,
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
            elif k == "normalize_world_space":
                data["normalize_world_space"] = v.lower() == "true"
    data.setdefault("data_factor", 1)
    data.setdefault("normalize_world_space", True)
    return data


def to_uint8_bgr(tensor):
    img = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def run_servo(mode, feature_builder, c2w_start, c2w_goal,
              means, quats, scales, opacities, colors,
              sh_degree, K_np, W, H, cam_params,
              mu, gain, max_iter, save_every, out_dir, device="cuda"):
    """Run VS with a given feature type."""
    os.makedirs(out_dir, exist_ok=True)

    cMo_start = np.linalg.inv(c2w_start)
    cMo_goal = np.linalg.inv(c2w_goal)

    # Desired
    rgb_des, gray_des, depth_des = render_gsplat(
        cMo_goal, means, quats, scales, opacities, colors,
        sh_degree, K_np, W, H, device)
    s_star = feature_builder()
    s_star.init(H, W)
    s_star.setCameraParameters(cam_params)
    s_star.buildFrom(gray_des, depth_des)

    # Camera simulator
    wMo = np.eye(4)
    robot = SimulatorCamera()
    robot.setPosition(wMo @ np.linalg.inv(cMo_start))
    robot.setRobotState(1)
    cMo = cMo_start.copy()

    print(f"\n[{mode}] Running...")
    print(f"{'Iter':>6} {'Error':>10} {'d(m)':>8} {'r(deg)':>8}")
    print("-" * 40)

    for it in range(max_iter):
        rgb_cur, gray_cur, depth_cur = render_gsplat(
            cMo, means, quats, scales, opacities, colors,
            sh_degree, K_np, W, H, device)

        s = feature_builder()
        s.init(H, W)
        s.setCameraParameters(cam_params)
        s.buildFrom(gray_cur, depth_cur)

        error = s.error(s_star)
        err_norm = torch.sum(error ** 2).item()
        c2w_cur = np.linalg.inv(cMo)
        t_err, r_err = se3_distance(c2w_cur, c2w_goal)

        if it % save_every == 0:
            # Save frame
            des_bgr = to_uint8_bgr(rgb_des)
            cur_bgr = to_uint8_bgr(rgb_cur)

            # For PGM-VS, show jet colormap of Gaussian mixture
            if hasattr(s, 'G_2d') and s.G_2d is not None:
                G_np = s.G_2d.cpu().numpy()
                G_vis = (G_np / (G_np.max() + 1e-8) * 255).clip(0, 255).astype(np.uint8)
                G_bgr = cv2.applyColorMap(G_vis, cv2.COLORMAP_JET)

                G_des_np = s_star.G_2d.cpu().numpy()
                G_des_vis = (G_des_np / (G_des_np.max() + 1e-8) * 255).clip(0, 255).astype(np.uint8)
                G_des_bgr = cv2.applyColorMap(G_des_vis, cv2.COLORMAP_JET)

                diff_g = np.abs(G_np - G_des_np)
                diff_vis = (diff_g / (diff_g.max() + 1e-8) * 255).clip(0, 255).astype(np.uint8)
                diff_bgr = cv2.applyColorMap(diff_vis, cv2.COLORMAP_HOT)

                Hf = G_des_bgr.shape[0]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(G_des_bgr, f"Desired PGM (lambda_g={s.lambda_g})", (5, 15), font, 0.4, (255,255,255), 1)
                cv2.putText(G_bgr, f"Current PGM (lambda_g={s.lambda_g})", (5, 15), font, 0.4, (255,255,255), 1)
                cv2.putText(diff_bgr, "|Diff| PGM", (5, 15), font, 0.4, (255,255,255), 1)
                cv2.putText(G_bgr, f"it={it} err={err_norm:.0f}", (5, Hf-25), font, 0.35, (255,255,255), 1)
                cv2.putText(G_bgr, f"t={t_err:.4f}m r={r_err:.1f}deg", (5, Hf-10), font, 0.35, (255,255,255), 1)
                concat = np.concatenate([G_des_bgr, G_bgr, diff_bgr], axis=1)
            # For DDVS, show the defocused grayscale images
            elif hasattr(s, 'I_d') and s.I_d is not None:
                I_d_np = s.I_d.cpu().numpy()
                I_d_vis = (I_d_np * 255).clip(0, 255).astype(np.uint8)
                I_d_bgr = cv2.cvtColor(I_d_vis, cv2.COLOR_GRAY2BGR)

                I_d_des_np = s_star.I_d.cpu().numpy()
                I_d_des_vis = (I_d_des_np * 255).clip(0, 255).astype(np.uint8)
                I_d_des_bgr = cv2.cvtColor(I_d_des_vis, cv2.COLOR_GRAY2BGR)

                diff_d = np.abs(I_d_np - I_d_des_np)
                diff_vis = (diff_d / (diff_d.max() + 1e-8) * 255).clip(0, 255).astype(np.uint8)
                diff_bgr = cv2.applyColorMap(diff_vis, cv2.COLORMAP_HOT)

                Hf = I_d_des_bgr.shape[0]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(I_d_des_bgr, "Desired (defocused)", (5, 15), font, 0.4, (255,255,255), 1)
                cv2.putText(I_d_bgr, "Current (defocused)", (5, 15), font, 0.4, (255,255,255), 1)
                cv2.putText(diff_bgr, "|Diff| DDVS", (5, 15), font, 0.4, (255,255,255), 1)
                cv2.putText(I_d_bgr, f"it={it} err={err_norm:.0f}", (5, Hf-25), font, 0.35, (255,255,255), 1)
                cv2.putText(I_d_bgr, f"t={t_err:.4f}m r={r_err:.1f}deg", (5, Hf-10), font, 0.35, (255,255,255), 1)
                concat = np.concatenate([I_d_des_bgr, I_d_bgr, diff_bgr], axis=1)
            else:
                diff = torch.clamp(torch.abs(rgb_des - rgb_cur) * 3, 0, 1)
                diff_bgr = to_uint8_bgr(diff)
                Hf = des_bgr.shape[0]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(des_bgr, "Desired", (5, 15), font, 0.4, (255,255,255), 1)
                cv2.putText(cur_bgr, "Current", (5, 15), font, 0.4, (255,255,255), 1)
                cv2.putText(diff_bgr, "|Diff|x3", (5, 15), font, 0.4, (255,255,255), 1)
                cv2.putText(cur_bgr, f"it={it} err={err_norm:.0f}", (5, Hf-25), font, 0.35, (255,255,255), 1)
                cv2.putText(cur_bgr, f"t={t_err:.4f}m r={r_err:.1f}deg", (5, Hf-10), font, 0.35, (255,255,255), 1)
                cv2.putText(cur_bgr, f"{mode}", (5, Hf-40), font, 0.35, (0,255,255), 1)
                concat = np.concatenate([des_bgr, cur_bgr, diff_bgr], axis=1)

            cv2.imwrite(os.path.join(out_dir, f"frame_{it:05d}.png"), concat)
            print(f"{it:>6} {err_norm:>10.0f} {t_err:>8.4f} {r_err:>8.2f}")

        # Convergence
        if t_err < 0.005 and r_err < 0.5:
            print(f"  CONVERGED at it={it}")
            break

        # Velocity (LM)
        Ls = s.interaction()
        Hs = Ls.T @ Ls
        diagHs = torch.diag(torch.diag(Hs))
        Hess = torch.linalg.inv(
            mu * diagHs + Hs + 1e-6 * torch.eye(6, device=device))
        v = -gain * (Hess @ Ls.T @ error)
        v_np = v.detach().cpu().numpy()

        vt = np.linalg.norm(v_np[:3])
        vr = np.linalg.norm(v_np[3:])
        if vt > 0.5: v_np[:3] *= 0.5 / vt
        if vr > 0.3: v_np[3:] *= 0.3 / vr

        robot.setVelocity("camera", v_np)
        wMc = robot.getPosition()
        cMo = np.linalg.inv(wMc) @ wMo

    else:
        print(f"  FAILED after {max_iter} iterations")

    n = len([f for f in os.listdir(out_dir) if f.startswith("frame_")])
    print(f"  Saved {n} frames to {out_dir}/")


def main():
    p = argparse.ArgumentParser(description="DDVS with 3DGS")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--data_factor", type=int, default=8)
    p.add_argument("--start_idx", type=int, default=14)
    p.add_argument("--goal_idx", type=int, default=10)

    # DDVS parameters
    p.add_argument("--aperture_phi", type=float, default=2.0,
                   help="F-number (smaller = more blur)")
    p.add_argument("--focus_depth", type=float, default=0.5,
                   help="Focus depth Z_f in meters")
    p.add_argument("--focal_length", type=float, default=0.017,
                   help="Lens focal length in meters")
    p.add_argument("--pixel_size", type=float, default=5.3e-6,
                   help="Physical pixel size in meters")

    # Inflated 3DGS parameters
    p.add_argument("--scale_factor", type=float, default=1.8)

    # PGM parameters
    p.add_argument("--pgm_lambda", type=float, default=10.0,
                   help="PGM Gaussian extension parameter lambda_g")

    # VS parameters
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--gain", type=float, default=10.0)
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--out_dir", type=str, default="logs/ddvs")
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

    c2w_start = camtoworlds[args.start_idx]
    c2w_goal = camtoworlds[args.goal_idx]
    init_t, init_r = se3_distance(c2w_start, c2w_goal)

    print(f"[Scene] {len(camtoworlds)} views, {W}x{H}")
    print(f"[Pair]  {args.start_idx} -> {args.goal_idx}: d={init_t:.3f}m, r={init_r:.1f}deg")
    print(f"[DDVS]  phi={args.aperture_phi}, Z_f={args.focus_depth}m")

    # Run original PVS
    run_servo("original",
              lambda: create_feature("pinhole", device=device, border=10),
              c2w_start, c2w_goal,
              means, quats, scales_original, opacities, colors,
              sh_degree, K_np, W, H, cam_params,
              args.mu, args.gain, args.max_iter, args.save_every,
              os.path.join(args.out_dir, "original"), device)

    # Run DDVS
    run_servo("ddvs",
              lambda: FeatureDDVS(
                  focal_length=args.focal_length,
                  aperture_phi=args.aperture_phi,
                  focus_depth=args.focus_depth,
                  pixel_size=args.pixel_size,
                  border=10, device=device),
              c2w_start, c2w_goal,
              means, quats, scales_original, opacities, colors,
              sh_degree, K_np, W, H, cam_params,
              args.mu, args.gain, args.max_iter, args.save_every,
              os.path.join(args.out_dir, "ddvs"), device)

    # Run inflated 3DGS
    run_servo("inflated",
              lambda: create_feature("pinhole", device=device, border=10),
              c2w_start, c2w_goal,
              means, quats, scales_original * args.scale_factor, opacities, colors,
              sh_degree, K_np, W, H, cam_params,
              args.mu, args.gain, args.max_iter, args.save_every,
              os.path.join(args.out_dir, "inflated"), device)

    # Run PGM-VS
    run_servo("pgm_vs",
              lambda: FeaturePGM(lambda_g=args.pgm_lambda, border=10, device=device),
              c2w_start, c2w_goal,
              means, quats, scales_original, opacities, colors,
              sh_degree, K_np, W, H, cam_params,
              args.mu, args.gain, args.max_iter, args.save_every,
              os.path.join(args.out_dir, "pgm_vs"), device)

    print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
