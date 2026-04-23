"""
Perspective vs Spherical Photometric Visual Servoing on 3DGS
=============================================================
v3 — Fixes: Z-depth to radial rho conversion, gradient scaling absorbed
     into L (not gradients), viser wxyz as numpy, velocity clamping.

Author: Youssef (UM6P / Ai Movement Lab)
"""

import argparse
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from gs_vs.simulators.RobotSimulator import SimulatorCamera
from gs_vs.simulators.CameraParameters import CameraParameters
from gs_vs.features.factory import create_feature
from gs_vs.tools.image_tools import save_rendered_images
from gs_vs.datasets.colmap import Parser
from gsplat.rendering import rasterization

import viser
import viser.transforms as vtf


# ==============================
# Parameters
# ==============================
mu = 0.01
lambda_ = 10.0
max_iter = 2000


# ==============================
# Utils
# ==============================

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


@torch.no_grad()
def render_gsplat(
    cMo, means, quats, scales, opacities, colors,
    sh_degree, K_np, W, H,
    camera_model="pinhole", device="cuda",
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


def pose_error(T_curr, T_des):
    dT = np.linalg.inv(T_des) @ T_curr
    t_err = np.linalg.norm(dT[:3, 3])
    cos_val = np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1)
    r_err = np.rad2deg(np.arccos(cos_val))
    return t_err, r_err


# ==============================
# Spherical Remapping
# ==============================

def build_perspective_to_sphere_maps(H, W, fx, fy, cx, cy):
    fov_h = 2 * np.arctan(W / (2 * fx))
    fov_v = 2 * np.arctan(H / (2 * fy))

    theta_range = np.linspace(-fov_h / 2, fov_h / 2, W)
    phi_range = np.linspace(-fov_v / 2, fov_v / 2, H)
    theta_grid, phi_grid = np.meshgrid(theta_range, phi_range)

    X = np.sin(theta_grid) * np.cos(phi_grid)
    Y = np.sin(phi_grid)
    Z = np.cos(theta_grid) * np.cos(phi_grid)
    Z = np.maximum(Z, 1e-8)

    map_x = (fx * X / Z + cx).astype(np.float32)
    map_y = (fy * Y / Z + cy).astype(np.float32)

    return map_x, map_y, theta_grid.astype(np.float32), phi_grid.astype(np.float32), fov_h, fov_v


def remap_to_sphere(gray_persp, map_x, map_y, device="cuda"):
    H, W = gray_persp.shape
    grid_x = torch.from_numpy(2.0 * map_x / (W - 1) - 1.0).to(device)
    grid_y = torch.from_numpy(2.0 * map_y / (H - 1) - 1.0).to(device)
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    img = gray_persp.unsqueeze(0).unsqueeze(0)
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border',
                         align_corners=True)[0, 0]


def remap_depth_to_sphere(depth_persp, map_x, map_y, device="cuda"):
    H, W = depth_persp.shape
    grid_x = torch.from_numpy(2.0 * map_x / (W - 1) - 1.0).to(device)
    grid_y = torch.from_numpy(2.0 * map_y / (H - 1) - 1.0).to(device)
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    img = depth_persp.unsqueeze(0).unsqueeze(0)
    # Use bilinear for smoother depth (nearest causes artifacts at edges)
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border',
                         align_corners=True)[0, 0]


def z_depth_to_radial(z_depth, theta_grid, phi_grid, device="cuda"):
    """
    Convert Z-depth (along optical axis) to radial distance rho.
    rho = Z / (cos(theta) * cos(phi))
    This is critical for the spherical interaction matrix.
    """
    theta = torch.from_numpy(theta_grid).to(device)
    phi = torch.from_numpy(phi_grid).to(device)
    cos_factor = torch.cos(theta) * torch.cos(phi)
    cos_factor = torch.clamp(cos_factor, min=0.1)  # safety
    return z_depth / cos_factor


# ==============================
# Image Gradients
# ==============================

def compute_image_gradients(gray, border=10):
    kx = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
                       dtype=torch.float32, device=gray.device) / 32.0
    ky = kx.T
    img = gray.unsqueeze(0).unsqueeze(0)
    Ix = F.conv2d(img, kx.unsqueeze(0).unsqueeze(0), padding=1)[0, 0]
    Iy = F.conv2d(img, ky.unsqueeze(0).unsqueeze(0), padding=1)[0, 0]
    Ix[:border, :] = 0; Ix[-border:, :] = 0; Ix[:, :border] = 0; Ix[:, -border:] = 0
    Iy[:border, :] = 0; Iy[-border:, :] = 0; Iy[:, :border] = 0; Iy[:, -border:] = 0
    return Ix, Iy


# ==============================
# Spherical Interaction Matrix
# ==============================

def interaction_matrix_spherical(Ix_pix, Iy_pix, rho, theta_grid, phi_grid,
                                  fov_h, fov_v, W, H, mask_s, device="cuda"):
    """
    Spherical photometric interaction matrix (Music et al. 2014).

    Uses PIXEL gradients (dI/du_s, dI/dv_s) and absorbs the pixel-to-radian
    conversion into L, rather than scaling gradients. This avoids numerical issues.

    L_Is = dI/du_s * (dtheta/du_s) * L_theta  +  dI/dv_s * (dphi/dv_s) * L_phi

    where dtheta/du_s = fov_h / W,  dphi/dv_s = fov_v / H

    rho must be RADIAL distance (not Z-depth).
    """
    theta = torch.from_numpy(theta_grid).to(device)
    phi = torch.from_numpy(phi_grid).to(device)
    rho_safe = torch.clamp(rho, min=0.01)

    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    sin_p, cos_p = torch.sin(phi), torch.cos(phi)
    tan_p = torch.tan(torch.clamp(phi, -1.4, 1.4))

    # Angular spacing per pixel
    dt_du = fov_h / W   # dtheta/du_s  (≈ 0.0003 for narrow FoV)
    dp_dv = fov_v / H   # dphi/dv_s

    # L_theta (6 cols) — scaled by dt_du
    Lt0 = dt_du * sin_t / rho_safe
    Lt1 = torch.zeros_like(Lt0)
    Lt2 = dt_du * (-cos_t) / rho_safe
    Lt3 = dt_du * sin_t * tan_p
    Lt4 = dt_du * (-torch.ones_like(Lt0))
    Lt5 = dt_du * (-cos_t * tan_p)

    # L_phi (6 cols) — scaled by dp_dv
    Lp0 = dp_dv * (-cos_t * sin_p) / rho_safe
    Lp1 = dp_dv * cos_p / rho_safe
    Lp2 = dp_dv * (-sin_t * sin_p) / rho_safe
    Lp3 = dp_dv * cos_t * cos_p
    Lp4 = torch.zeros_like(Lp0)
    Lp5 = dp_dv * (-sin_t * cos_p)

    # L_Is = dI/du * (dt_du * L_theta) + dI/dv * (dp_dv * L_phi)
    L0 = Ix_pix * Lt0 + Iy_pix * Lp0
    L1 = Ix_pix * Lt1 + Iy_pix * Lp1
    L2 = Ix_pix * Lt2 + Iy_pix * Lp2
    L3 = Ix_pix * Lt3 + Iy_pix * Lp3
    L4 = Ix_pix * Lt4 + Iy_pix * Lp4
    L5 = Ix_pix * Lt5 + Iy_pix * Lp5

    L_full = torch.stack([L0, L1, L2, L3, L4, L5], dim=-1)
    return L_full[mask_s]


# ==============================
# Perspective PVS
# ==============================

@torch.no_grad()
def run_perspective_pvs(
    means, quats, scales, opacities, colors, sh_degree,
    K_np, W, H, cam_params,
    cdMo, cMo_init, feature_type, camera_model, device="cuda"
):
    _, gray_des, depth_des, _ = render_gsplat(
        cdMo, means, quats, scales, opacities, colors,
        sh_degree, K_np, W, H, camera_model=camera_model)

    s_star = create_feature(feature_type, device="cuda", border=10)
    s_star.init(H, W)
    s_star.setCameraParameters(cam_params)
    s_star.buildFrom(gray_des, depth_des, mask=None)

    s = create_feature(feature_type, device="cuda", border=10)
    s.init(H, W)
    s.setCameraParameters(cam_params)

    robot = SimulatorCamera()
    wMo = np.eye(4)
    cMo = cMo_init.copy()
    robot.setPosition(wMo @ np.linalg.inv(cMo))
    robot.setRobotState(1)

    trajectory = [cMo.copy()]
    err_history = []

    for it in range(max_iter):
        _, gray, depth, _ = render_gsplat(
            cMo, means, quats, scales, opacities, colors,
            sh_degree, K_np, W, H, camera_model=camera_model)

        s.buildFrom(gray, depth, mask=None)
        error = s.error(s_star)
        err_norm = torch.sum(error ** 2).item()
        err_history.append(err_norm)

        Ls = s.interaction()
        Hs = Ls.T @ Ls
        diagHs = torch.diag(torch.diag(Hs))
        Hess = torch.linalg.inv(mu * diagHs + Hs + 1e-6 * torch.eye(6, device=device))
        v = -lambda_ * (Hess @ Ls.T @ error)

        v_np = v.detach().cpu().numpy()
        robot.setVelocity("camera", v_np)
        wMc = robot.getPosition()
        cMo = np.linalg.inv(wMc) @ wMo
        s.reset()
        trajectory.append(cMo.copy())

        if it % 100 == 0:
            t_err, r_err = pose_error(cMo, cdMo)
            print(f"  [PERSP] iter {it:4d} | SSD={err_norm:.1f} | t={t_err:.4f}m r={r_err:.2f}°")
        if err_norm < 1e2:
            print(f"  [PERSP] Converged at iter {it}, err={err_norm:.1f}")
            break

    return trajectory, err_history


# ==============================
# Spherical PVS
# ==============================

@torch.no_grad()
def run_spherical_pvs(
    means, quats, scales, opacities, colors, sh_degree,
    K_np, W, H, cdMo, cMo_init, camera_model, device="cuda"
):
    fx, fy = K_np[0, 0], K_np[1, 1]
    cx, cy = K_np[0, 2], K_np[1, 2]

    map_x, map_y, theta_grid, phi_grid, fov_h, fov_v = \
        build_perspective_to_sphere_maps(H, W, fx, fy, cx, cy)

    print(f"  [SPHER] FoV: {np.rad2deg(fov_h):.1f}° x {np.rad2deg(fov_v):.1f}°")
    print(f"  [SPHER] Angular spacing: dtheta/du={fov_h/W:.6f} rad/px, dphi/dv={fov_v/H:.6f} rad/px")

    border = 10
    mask_s = torch.ones(H, W, dtype=torch.bool, device=device)
    mask_s[:border, :] = False; mask_s[-border:, :] = False
    mask_s[:, :border] = False; mask_s[:, -border:] = False

    # Desired (spherical)
    _, gray_des_p, depth_des_p, _ = render_gsplat(
        cdMo, means, quats, scales, opacities, colors,
        sh_degree, K_np, W, H, camera_model=camera_model)
    gray_des_s = remap_to_sphere(gray_des_p, map_x, map_y, device)

    robot = SimulatorCamera()
    wMo = np.eye(4)
    cMo = cMo_init.copy()
    robot.setPosition(wMo @ np.linalg.inv(cMo))
    robot.setRobotState(1)

    trajectory = [cMo.copy()]
    err_history = []

    for it in range(max_iter):
        _, gray_cur_p, depth_cur_p, _ = render_gsplat(
            cMo, means, quats, scales, opacities, colors,
            sh_degree, K_np, W, H, camera_model=camera_model)

        gray_cur_s = remap_to_sphere(gray_cur_p, map_x, map_y, device)
        depth_z_s = remap_depth_to_sphere(depth_cur_p, map_x, map_y, device)

        # CRITICAL: convert Z-depth to radial distance rho
        rho = z_depth_to_radial(depth_z_s, theta_grid, phi_grid, device)

        # Error
        error = (gray_cur_s - gray_des_s)[mask_s]
        err_norm = torch.sum(error ** 2).item()
        err_history.append(err_norm)

        # Pixel gradients (NOT scaled — scaling is inside L)
        Ix_pix, Iy_pix = compute_image_gradients(gray_cur_s, border=border)

        # Spherical interaction matrix (with angular scaling absorbed)
        Ls = interaction_matrix_spherical(
            Ix_pix, Iy_pix, rho, theta_grid, phi_grid,
            fov_h, fov_v, W, H, mask_s, device)

        # LM velocity
        Hs = Ls.T @ Ls
        diagHs = torch.diag(torch.diag(Hs))
        Hess = torch.linalg.inv(mu * diagHs + Hs + 1e-6 * torch.eye(6, device=device))
        v = -lambda_ * (Hess @ Ls.T @ error)

        v_np = v.detach().cpu().numpy()

        # Clamp velocity for stability
        v_np[:3] = np.clip(v_np[:3], -0.05, 0.05)
        v_np[3:] = np.clip(v_np[3:], -0.1, 0.1)

        robot.setVelocity("camera", v_np)
        wMc = robot.getPosition()
        cMo = np.linalg.inv(wMc) @ wMo
        trajectory.append(cMo.copy())

        if it % 100 == 0:
            t_err, r_err = pose_error(cMo, cdMo)
            v_lin = np.linalg.norm(v_np[:3])
            v_ang = np.linalg.norm(v_np[3:])
            print(f"  [SPHER] iter {it:4d} | SSD={err_norm:.1f} | t={t_err:.4f}m r={r_err:.2f}° | "
                  f"v_lin={v_lin:.5f} v_ang={v_ang:.5f}")
        if err_norm < 1e2:
            print(f"  [SPHER] Converged at iter {it}, err={err_norm:.1f}")
            break

    return trajectory, err_history


# ==============================
# Viser Comparison Viewer
# ==============================

def visualize_comparison(traj_persp, traj_sphere, err_persp, err_sphere,
                         cMo_init, cdMo, means_np, port=8080):
    server = viser.ViserServer(host="0.0.0.0", port=port)

    def poses_to_positions(traj):
        return np.array([np.linalg.inv(T)[:3, 3] for T in traj]).astype(np.float32)

    pos_p = poses_to_positions(traj_persp)
    pos_s = poses_to_positions(traj_sphere)

    # Scene points
    if means_np is not None:
        idx = np.random.choice(len(means_np), min(50000, len(means_np)), replace=False)
        server.scene.add_point_cloud("/scene/pts", points=means_np[idx].astype(np.float32),
            colors=np.full((len(idx), 3), 180, dtype=np.uint8), point_size=0.003)

    # Trajectories
    for name, pos, color in [("perspective", pos_p, [220, 50, 50]),
                              ("spherical", pos_s, [50, 100, 220])]:
        if len(pos) > 1:
            pts = np.stack([pos[:-1], pos[1:]], axis=1).astype(np.float32)
            server.scene.add_line_segments(f"/traj/{name}", points=pts,
                colors=np.array(color, dtype=np.uint8), line_width=3.0)

    # Ideal line
    p_start = np.linalg.inv(cMo_init)[:3, 3]
    p_goal = np.linalg.inv(cdMo)[:3, 3]
    pts_i = np.stack([p_start, p_goal]).reshape(1, 2, 3).astype(np.float32)
    server.scene.add_line_segments("/traj/ideal", points=pts_i,
        colors=np.array([150, 150, 150], dtype=np.uint8), line_width=2.0)

    # Frustums — wxyz must be numpy array, not list!
    for name, T_cMo, color in [("start", cMo_init, (255, 200, 0)), ("goal", cdMo, (0, 255, 0))]:
        T_wMc = np.linalg.inv(T_cMo)
        q = Rot.from_matrix(T_wMc[:3, :3]).as_quat()  # [x, y, z, w]
        wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
        server.scene.add_camera_frustum(f"/markers/{name}", fov=np.deg2rad(60), aspect=4/3,
            scale=0.08, wxyz=wxyz, position=T_wMc[:3, 3].astype(np.float64), color=color)

    # Metrics
    def path_len(pos):
        if len(pos) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))

    lp, ls, li = path_len(pos_p), path_len(pos_s), float(np.linalg.norm(p_goal - p_start))
    t_fp, r_fp = pose_error(traj_persp[-1], cdMo)
    t_fs, r_fs = pose_error(traj_sphere[-1], cdMo)

    with server.gui.add_folder("Results"):
        server.gui.add_markdown(
            f"**Path Length:** ideal={li:.4f}m, persp={lp:.4f}m ({lp/max(li,1e-6):.2f}x), "
            f"spher={ls:.4f}m ({ls/max(li,1e-6):.2f}x)\n\n"
            f"**Straightness** (1.0=perfect): persp={li/max(lp,1e-6):.3f}, spher={li/max(ls,1e-6):.3f}\n\n"
            f"**Iterations:** persp={len(err_persp)}, spher={len(err_sphere)}\n\n"
            f"**Final Error:** persp: t={t_fp:.4f}m r={r_fp:.2f} | spher: t={t_fs:.4f}m r={r_fs:.2f}"
        )

    with server.gui.add_folder("Legend"):
        server.gui.add_markdown(
            "**Red** = Perspective PVS (classical)\n\n"
            "**Blue** = Spherical PVS (decoupled)\n\n"
            "**Gray** = Ideal straight line\n\n"
            "**Yellow** = Start, **Green** = Goal"
        )

    server.scene.add_frame("/origin", axes_length=0.2, axes_radius=0.008)
    print(f"\n  VIEWER: http://localhost:{port}")
    return server


# ==============================
# Main
# ==============================

@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_basic_cfg_fields(args.cfg)
    parser = Parser(data_dir=cfg["data_dir"], factor=cfg["data_factor"],
                    normalize=cfg["normalize_world_space"], test_every=8)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    splats = ckpt["splats"]
    means = splats["means"].to(device)
    quats = splats["quats"].to(device)
    scales = torch.exp(splats["scales"]).to(device)
    opacities = torch.sigmoid(splats["opacities"]).to(device)
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1).to(device)
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)

    camtoworlds = torch.from_numpy(parser.camtoworlds).float().to(device)

    W, H = list(parser.imsize_dict.values())[0]
    K_colmap = torch.from_numpy(list(parser.Ks_dict.values())[0]).float()
    fx, fy = K_colmap[0, 0].item(), K_colmap[1, 1].item()
    cx, cy = K_colmap[0, 2].item(), K_colmap[1, 2].item()
    K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    cam_params = CameraParameters(px=fx, py=fy, u0=cx, v0=cy)

    print(f"  Resolution: {W}x{H}, fx={fx:.1f}, fy={fy:.1f}")
    print(f"  Gaussians: {len(means)}, sh_degree={sh_degree}")

    # Desired pose
    idx = args.desired_image_index
    cdMo = np.linalg.inv(camtoworlds[idx].cpu().numpy())

    # Initial pose = desired + displacement
    cMo_init = cdMo.copy()
    dt = np.array(args.delta_t)
    dr = np.deg2rad(args.delta_r)
    dR = Rot.from_rotvec(dr).as_matrix()
    cMo_init[:3, :3] = dR @ cMo_init[:3, :3]
    cMo_init[:3, 3] += dt

    t0_err, r0_err = pose_error(cMo_init, cdMo)
    print(f"  Initial displacement: {t0_err:.4f}m, {r0_err:.2f}°")

    # --- Perspective PVS ---
    print(f"\n{'='*40}\n  PERSPECTIVE PVS\n{'='*40}")
    t0 = time.time()
    traj_p, err_p = run_perspective_pvs(
        means, quats, scales, opacities, colors, sh_degree,
        K_np, W, H, cam_params, cdMo, cMo_init,
        args.feature_type, args.camera_model, device)
    time_p = time.time() - t0

    # --- Spherical PVS ---
    print(f"\n{'='*40}\n  SPHERICAL PVS\n{'='*40}")
    t0 = time.time()
    traj_s, err_s = run_spherical_pvs(
        means, quats, scales, opacities, colors, sh_degree,
        K_np, W, H, cdMo, cMo_init, args.camera_model, device)
    time_s = time.time() - t0

    # --- Summary ---
    tfp, rfp = pose_error(traj_p[-1], cdMo)
    tfs, rfs = pose_error(traj_s[-1], cdMo)
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  {'':20s} {'Perspective':>14s} {'Spherical':>14s}")
    print(f"  {'Iterations':20s} {len(err_p):14d} {len(err_s):14d}")
    print(f"  {'Final t_err (m)':20s} {tfp:14.4f} {tfs:14.4f}")
    print(f"  {'Final r_err (deg)':20s} {rfp:14.2f} {rfs:14.2f}")
    print(f"  {'Time (s)':20s} {time_p:14.1f} {time_s:14.1f}")
    print(f"  {'Final SSD':20s} {err_p[-1]:14.1f} {err_s[-1]:14.1f}")
    print(f"{'='*60}")

    # --- Viewer ---
    server = visualize_comparison(traj_p, traj_s, err_p, err_s,
                                   cMo_init, cdMo, means.cpu().numpy(), args.port)
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--camera_model", default="pinhole", choices=["pinhole", "fisheye"])
    p.add_argument("--feature_type", default="pinhole",
                   choices=["pinhole", "unified_ip", "unified_cs", "unified_ps", "equidistant"])
    p.add_argument("--desired_image_index", default=0, type=int)
    p.add_argument("--delta_t", nargs=3, type=float, default=[0.05, 0.05, -0.05])
    p.add_argument("--delta_r", nargs=3, type=float, default=[0.0, 5.0, 0.0])
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"  PERSPECTIVE vs SPHERICAL PVS COMPARISON")
    print(f"{'='*60}")
    main(args)
