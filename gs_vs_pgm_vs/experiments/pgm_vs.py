"""
Visual Servoing on Gaussian Splatting scenes.

Supports three modes:
  --mode pgm   : PGM-VS only (Crombez et al. TRO 2019)
  --mode pl    : Traditional photometric VS only (Collewet & Marchand TRO 2011)
  --mode both  : Side-by-side comparison

Results are visualized in a Viser viewer with 3D trajectories.
"""

import argparse
import sys
import os
import time
import torch
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gs_vs.simulators.RobotSimulator import SimulatorCamera
from gs_vs.simulators.CameraParameters import CameraParameters
from gs_vs.features.factory import create_feature
from gs_vs.tools.image_tools import save_rendered_images, compute_fisheye_mask_v2
from gs_vs.datasets.colmap import Parser
from gsplat.rendering import rasterization

from gs_vs_pgm_vs.features.FeaturePGM import FeaturePGM
from gs_vs_pgm_vs.viewers.viewer_comparison import VsComparisonViewer


# ==============================
# Utils
# ==============================
@torch.no_grad()
def render_gsplat(
    cMo, means, quats, scales, opacities, colors,
    sh_degree, K_np, W, H,
    camera_model="pinhole",
    device="cuda",
):
    viewmat = torch.from_numpy(cMo).float().to(device)[None]
    Ks = torch.from_numpy(K_np).float().to(device)[None]

    renders, render_alphas, _ = rasterization(
        means=means, quats=quats, scales=scales,
        opacities=opacities, colors=colors,
        sh_degree=sh_degree, viewmats=viewmat, Ks=Ks,
        width=W, height=H, packed=True,
        render_mode="RGB+ED", camera_model=camera_model,
    )

    rgb = torch.clamp(renders[0, ..., :3], 0.0, 1.0)
    depth = renders[0, ..., 3]
    gray = (0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]) * 255.0
    mask = (render_alphas[0, ..., 0] > 1e-4)
    return rgb, gray, depth, mask


def load_basic_cfg_fields(cfg_path):
    data = {}
    with open(cfg_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if k == "data_dir":
                data["data_dir"] = v
            elif k == "data_factor":
                data["data_factor"] = int(v)
            elif k == "normalize_world_space":
                data["normalize_world_space"] = v.lower() == "true"
    data.setdefault("data_factor", 1)
    data.setdefault("normalize_world_space", True)
    return data


def apply_displacement(cMo, displacement_str):
    cMo_out = cMo.copy()
    disp = np.array([float(x) for x in displacement_str.split(',')])
    cMo_out[:3, 3] += disp[:3]
    if len(disp) > 3 and np.any(disp[3:6] != 0):
        angles_rad = np.deg2rad(disp[3:6])
        R_disp = R.from_euler('xyz', angles_rad).as_matrix()
        cMo_out[:3, :3] = R_disp @ cMo_out[:3, :3]
    return cMo_out


def compute_pose_error(cMo_current, cMo_desired):
    pose_err = cMo_current @ np.linalg.inv(cMo_desired)
    t_err = np.linalg.norm(pose_err[:3, 3]) * 1000.0
    r_err = np.linalg.norm(R.from_matrix(pose_err[:3, :3]).as_rotvec()) * 180.0 / np.pi
    return t_err, r_err


# ==============================
# PL-VS step (traditional photometric)
# ==============================
def pl_step(s_pl, s_star_pl, mu_pl, lambda_pl, device):
    error = s_pl.error(s_star_pl)
    err_sse = torch.sum(error ** 2).item()
    n_pixels = error.shape[0]

    Ls = s_pl.interaction()
    Hs = Ls.T @ Ls
    diagHs = torch.diag(torch.diag(Hs))
    Hess = torch.linalg.inv(mu_pl * diagHs + Hs + 1e-6 * torch.eye(6, device=device))
    v = -lambda_pl * (Hess @ Ls.T @ error)

    return v, err_sse, n_pixels


# ==============================
# PGM-VS step
# ==============================
def pgm_step(s_pgm, s_star_pgm, mu_pgm, device):
    error = s_pgm.error(s_star_pgm)
    err_sse = torch.sum(error ** 2).item()
    n_pixels = error.shape[0]

    LG_ext = s_pgm.interaction_extended()
    LtL = LG_ext.T @ LG_ext
    damping = 0.01 * torch.diag(torch.diag(LtL)) + 1e-6 * torch.eye(7, device=device)
    v_lambda = -mu_pgm * torch.linalg.solve(LtL + damping, LG_ext.T @ error)

    v_cam = v_lambda[:6]
    delta_lambda = v_lambda[6].item()

    return v_cam, delta_lambda, err_sse, n_pixels


# ==============================
# Main
# ==============================
@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mode = args.mode  # "pgm", "pl", or "both"
    run_pgm = mode in ("pgm", "both")
    run_pl = mode in ("pl", "both")

    # ---- Load config & data ----
    cfg = load_basic_cfg_fields(args.cfg)
    colmap_parser = Parser(
        data_dir=cfg["data_dir"],
        factor=cfg["data_factor"],
        normalize=cfg["normalize_world_space"],
        test_every=8,
    )

    # === Load GS checkpoint ===
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    splats = ckpt["splats"]
    means = splats["means"].to(device)
    quats = splats["quats"].to(device)
    scales = torch.exp(splats["scales"]).to(device)
    opacities = torch.sigmoid(splats["opacities"]).to(device)
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1).to(device)
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)

    camtoworlds = torch.from_numpy(colmap_parser.camtoworlds).float().to(device)

    # === Camera intrinsics ===
    if args.intrinsics_file is not None:
        print(f"[INFO] Loading intrinsics from: {args.intrinsics_file}")
        with open(args.intrinsics_file, "r") as f:
            intr = yaml.safe_load(f)
        W, H = intr["width"], intr["height"]
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    else:
        print("[INFO] Using intrinsics from COLMAP parser")
        W, H = list(colmap_parser.imsize_dict.values())[0]
        K_colmap = torch.from_numpy(list(colmap_parser.Ks_dict.values())[0]).float()
        fx, fy = K_colmap[0, 0].item(), K_colmap[1, 1].item()
        cx, cy = K_colmap[0, 2].item(), K_colmap[1, 2].item()

    K_np = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    apply_mask_flag = (args.camera_model == "fisheye")
    fisheye_mask = None
    if apply_mask_flag:
        fisheye_mask = compute_fisheye_mask_v2(W, H, cx, cy)

    # === Desired pose & image ===
    idx = int(args.desired_image_index)
    print(f"[INFO] Desired view index: {idx}")
    camtoworld = camtoworlds[idx].cpu().numpy()
    cdMo = np.linalg.inv(camtoworld)

    rgb_des, gray_des, depth_des, _ = render_gsplat(
        cdMo, means, quats, scales, opacities, colors,
        sh_degree, K_np, W, H, camera_model=args.camera_model
    )
    if apply_mask_flag:
        rgb_des[~fisheye_mask] = 0
        gray_des[~fisheye_mask] = 0
        depth_des[~fisheye_mask] = 0

    os.makedirs("logs/pgm_vs", exist_ok=True)
    save_rendered_images(rgb_des, gray_des, depth_des, mask=fisheye_mask,
                         out_dir="logs/pgm_vs", prefix="des")

    cam_params = CameraParameters(px=fx, py=fy, u0=cx, v0=cy)

    # === Initial pose ===
    cMo_init = apply_displacement(cdMo, args.displacement)

    rgb_ini, gray_ini, depth_ini, _ = render_gsplat(
        cMo_init, means, quats, scales, opacities, colors,
        sh_degree, K_np, W, H, camera_model=args.camera_model
    )
    if apply_mask_flag:
        rgb_ini[~fisheye_mask] = 0
        gray_ini[~fisheye_mask] = 0
        depth_ini[~fisheye_mask] = 0

    save_rendered_images(rgb_ini, gray_ini, depth_ini, mask=fisheye_mask,
                         out_dir="logs/pgm_vs", prefix="ini")

    wMo = np.eye(4)

    # ============================================================
    # Setup PGM-VS (if needed)
    # ============================================================
    if run_pgm:
        lambda_gi = args.lambda_gi
        lambda_g_cur = lambda_gi
        lambda_g_star = lambda_gi / 2.0
        lambda_g_final = args.lambda_g_final
        switch_threshold = args.switch_threshold
        pgm_step_num = 1

        s_star_pgm = FeaturePGM(lambda_g=lambda_g_star, border=10, device=device)
        s_star_pgm.init(H, W)
        s_star_pgm.setCameraParameters(cam_params)
        s_star_pgm.buildFrom(gray_des, depth_des, mask=fisheye_mask)

        s_pgm = FeaturePGM(lambda_g=lambda_g_cur, border=10, device=device)
        s_pgm.init(H, W)
        s_pgm.setCameraParameters(cam_params)

        cMo_pgm = cMo_init.copy()
        robot_pgm = SimulatorCamera()
        robot_pgm.setPosition(wMo @ np.linalg.inv(cMo_pgm))
        robot_pgm.setRobotState(1)

    # ============================================================
    # Setup PL-VS (if needed)
    # ============================================================
    if run_pl:
        feature_type_pl = args.feature_type_pl

        s_star_pl = create_feature(feature_type_pl, device=device, border=10)
        s_star_pl.init(H, W)
        s_star_pl.setCameraParameters(cam_params)
        s_star_pl.buildFrom(gray_des, depth_des, mask=fisheye_mask)

        s_pl = create_feature(feature_type_pl, device=device, border=10)
        s_pl.init(H, W)
        s_pl.setCameraParameters(cam_params)

        cMo_pl = cMo_init.copy()
        robot_pl = SimulatorCamera()
        robot_pl.setPosition(wMo @ np.linalg.inv(cMo_pl))
        robot_pl.setRobotState(1)

    # ============================================================
    # Viewer
    # ============================================================
    # For single-method modes, pass the same pose for the unused method
    # (its frustum will be hidden)
    viewer = VsComparisonViewer(
        rgb_des=rgb_des.cpu().numpy(),
        cdMo=np.linalg.inv(cdMo),
        rgb_cur_pgm=rgb_ini.cpu().numpy(),
        cMo_pgm=np.linalg.inv(cMo_init),
        rgb_cur_pl=rgb_ini.cpu().numpy(),
        cMo_pl=np.linalg.inv(cMo_init),
        image_size=(W, H),
        aspect_ratio=W / H,
        server_port=8080,
        image_scale=0.5,
        means=means.detach().cpu().numpy(),
        scales=scales.detach().cpu().numpy(),
        quats=quats.detach().cpu().numpy(),
        sh0=splats["sh0"].detach().cpu().numpy(),
        opacities=opacities.detach().cpu().numpy(),
        enable_pgm=run_pgm,
        enable_pl=run_pl,
    )

    # ============================================================
    # Control parameters
    # ============================================================
    mu_pgm = args.gain_pgm
    mu_pl = args.mu_pl
    lambda_pl = args.lambda_pl
    max_iter = args.max_iter
    conv_thresh = args.convergence_threshold

    pgm_converged = not run_pgm   # skip if not running
    pl_converged = not run_pl

    err_pgm_mse = 0.0
    err_pl_mse = 0.0

    print(f"\n{'='*60}")
    print(f"Mode: {mode.upper()} (max {max_iter} iterations)")
    if run_pgm:
        print(f"  PGM-VS: gain={mu_pgm}, lambda_gi={args.lambda_gi}")
    if run_pl:
        print(f"  PL-VS:  mu={mu_pl}, lambda={lambda_pl}, feature={args.feature_type_pl}")
    print(f"{'='*60}\n")

    # ============================================================
    # Servo Loop
    # ============================================================
    for it in range(max_iter):

        # ==========================
        # PGM-VS iteration
        # ==========================
        if run_pgm and not pgm_converged:
            rgb_pgm, gray_pgm, depth_pgm, _ = render_gsplat(
                cMo_pgm, means, quats, scales, opacities, colors,
                sh_degree, K_np, W, H, camera_model=args.camera_model
            )
            if apply_mask_flag:
                rgb_pgm[~fisheye_mask] = 0
                gray_pgm[~fisheye_mask] = 0
                depth_pgm[~fisheye_mask] = 0

            s_pgm.setLambda(lambda_g_cur)
            s_pgm.buildFrom(gray_pgm, depth_pgm, mask=fisheye_mask)

            v_pgm, delta_lam, err_pgm, n_pgm = pgm_step(s_pgm, s_star_pgm, mu_pgm, device)
            err_pgm_mse = err_pgm / max(n_pgm, 1)

            lambda_g_cur = max(lambda_g_cur + delta_lam, 0.1)

            if pgm_step_num == 1 and abs(lambda_g_cur - lambda_g_star) < switch_threshold:
                pgm_step_num = 2
                lambda_g_cur = lambda_g_final
                lambda_g_star = lambda_g_final
                s_star_pgm.setLambda(lambda_g_star)
                s_star_pgm.buildFrom(gray_des, depth_des, mask=fisheye_mask)
                print(f"  [PGM] Step 1->2 at iter {it}, lambda_g={lambda_g_final}")

            v_pgm_np = v_pgm.detach().cpu().numpy()
            robot_pgm.setVelocity("camera", v_pgm_np)
            wMc_pgm = robot_pgm.getPosition()
            cMo_pgm = np.linalg.inv(wMc_pgm) @ wMo
            s_pgm.reset()

            viewer.update_pgm(
                it, np.linalg.inv(cMo_pgm), rgb_pgm,
                err_pgm_mse, v_pgm_np,
                lambda_g=lambda_g_cur, lambda_g_star=lambda_g_star
            )

            if err_pgm_mse < conv_thresh:
                pgm_converged = True
                t_e, r_e = compute_pose_error(cMo_pgm, cdMo)
                print(f"  [PGM] Converged at iter {it} | t={t_e:.2f}mm, r={r_e:.3f}deg")

        # ==========================
        # PL-VS iteration
        # ==========================
        if run_pl and not pl_converged:
            rgb_pl, gray_pl, depth_pl, _ = render_gsplat(
                cMo_pl, means, quats, scales, opacities, colors,
                sh_degree, K_np, W, H, camera_model=args.camera_model
            )
            if apply_mask_flag:
                rgb_pl[~fisheye_mask] = 0
                gray_pl[~fisheye_mask] = 0
                depth_pl[~fisheye_mask] = 0

            s_pl.buildFrom(gray_pl, depth_pl, mask=fisheye_mask)

            v_pl, err_pl, n_pl = pl_step(s_pl, s_star_pl, mu_pl, lambda_pl, device)
            err_pl_mse = err_pl / max(n_pl, 1)

            v_pl_np = v_pl.detach().cpu().numpy()
            robot_pl.setVelocity("camera", v_pl_np)
            wMc_pl = robot_pl.getPosition()
            cMo_pl = np.linalg.inv(wMc_pl) @ wMo
            s_pl.reset()

            viewer.update_pl(
                it, np.linalg.inv(cMo_pl), rgb_pl,
                err_pl_mse, v_pl_np
            )

            if err_pl_mse < conv_thresh:
                pl_converged = True
                t_e, r_e = compute_pose_error(cMo_pl, cdMo)
                print(f"  [PL]  Converged at iter {it} | t={t_e:.2f}mm, r={r_e:.3f}deg")

        # Refresh viewer
        viewer.refresh_plots(iteration=it)

        # Print progress
        if it % 50 == 0 or it < 10:
            parts = [f"iter {it:4d}"]
            if run_pgm:
                pgm_s = f"mse={err_pgm_mse:.2e}, lam={lambda_g_cur:.1f}" if not pgm_converged else "CONVERGED"
                parts.append(f"PGM: {pgm_s}")
            if run_pl:
                pl_s = f"mse={err_pl_mse:.2e}" if not pl_converged else "CONVERGED"
                parts.append(f"PL: {pl_s}")
            print("  " + " | ".join(parts))

        time.sleep(0.05)

        if pgm_converged and pl_converged:
            print(f"\nAll active methods converged at iteration {it}.")
            break

    # ============================================================
    # Final results
    # ============================================================
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    if run_pgm:
        t_pgm, r_pgm = compute_pose_error(cMo_pgm, cdMo)
        print(f"  PGM-VS: t={t_pgm:.2f}mm, r={r_pgm:.3f}deg | converged={pgm_converged}")
    if run_pl:
        t_pl, r_pl = compute_pose_error(cMo_pl, cdMo)
        print(f"  PL-VS:  t={t_pl:.2f}mm, r={r_pl:.3f}deg | converged={pl_converged}")

    print(f"{'='*60}\n")

    viewer.save("logs/pgm_vs/recording.viser")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual Servoing on Gaussian Splatting (PGM / PL / both)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode
    parser.add_argument("--mode", default="both", choices=["pgm", "pl", "both"],
                        help="Which method(s) to run: pgm, pl, or both")

    # Required
    parser.add_argument("--ckpt", required=True, help="GS checkpoint path")
    parser.add_argument("--cfg", required=True, help="GS config path")

    # Camera
    parser.add_argument("--camera_model", default="pinhole", choices=["pinhole", "fisheye"])
    parser.add_argument("--intrinsics_file", default=None)
    parser.add_argument("--desired_image_index", default=0)

    # PGM parameters
    parser.add_argument("--lambda_gi", type=float, default=25.0,
                        help="Initial Gaussian extension parameter")
    parser.add_argument("--lambda_g_final", type=float, default=1.0,
                        help="Final lambda_g for accuracy")
    parser.add_argument("--switch_threshold", type=float, default=0.1,
                        help="Step 1->2 transition threshold")
    parser.add_argument("--gain_pgm", type=float, default=1.0,
                        help="PGM control gain (mu)")

    # PL parameters
    parser.add_argument("--feature_type_pl", default="pinhole",
                        choices=["pinhole", "unified_ip", "unified_cs", "unified_ps", "equidistant"],
                        help="Feature type for PL-VS")
    parser.add_argument("--mu_pl", type=float, default=0.01,
                        help="PL Levenberg-Marquardt damping")
    parser.add_argument("--lambda_pl", type=float, default=10.0,
                        help="PL control gain")

    # Common
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--convergence_threshold", type=float, default=10.0,
                        help="Convergence on MSE (||e||^2 / n_pixels)")
    parser.add_argument("--displacement", type=str, default="0.2,0.2,-0.2,0,0,0",
                        help="tx,ty,tz,rx,ry,rz (meters,degrees)")

    args = parser.parse_args()

    mode_labels = {"pgm": "PGM-VS only", "pl": "PL-VS only", "both": "PGM-VS vs PL-VS"}
    print("\n" + "=" * 60)
    print(f"VISUAL SERVOING ON GAUSSIAN SPLATTING")
    print(f"Mode: {mode_labels[args.mode]}")
    print("=" * 60)
    print(f"Checkpoint:    {args.ckpt}")
    print(f"Camera:        {args.camera_model}")
    if args.mode in ("pgm", "both"):
        print(f"PGM lambda_gi: {args.lambda_gi}")
    if args.mode in ("pl", "both"):
        print(f"PL feature:    {args.feature_type_pl}")
    print(f"Displacement:  {args.displacement}")
    print("=" * 60 + "\n")

    main(args)
