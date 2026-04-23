"""
UR10e Photometric Visual Servoing with 3D Gaussian Splatting
=============================================================

Eye-in-hand VS simulation: camera on UR10e end-effector, renders via gsplat.

The camera motion is computed by the VS control law (same as free-flying),
then mapped to joint space via the robot Jacobian for visualization.
This shows that the VS velocities are compatible with the robot kinematics.

Renders both camera view [desired|current|diff] and logs joint trajectories.

Usage:
    python -m gs_vs_scaling_gaussians.ur5_simulation.ur5_pvs_gsplat \
        --ckpt <checkpoint> --cfg <config> \
        --goal_idx 10 --start_idx 12 \
        --modes original inflated \
        --out_dir logs/ur10e_sim
"""

import argparse
import os
import torch
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot

from gs_vs.simulators.RobotSimulator import SimulatorCamera
from gs_vs.simulators.CameraParameters import CameraParameters
from gs_vs.features.factory import create_feature
from gs_vs.datasets.colmap import Parser
from gsplat.rendering import rasterization

from gs_vs_scaling_gaussians.ur5_simulation.ur5_kinematics import UR_Simulator
from gs_vs_scaling_gaussians.ur5_simulation.mujoco_renderer import MuJoCoUR10eRenderer


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


def to_uint8_bgr(tensor):
    img = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def make_frame(rgb_des, rgb_cur, it, err, t_err, r_err, q_deg, mode, sf,
               mj_frame=None):
    """
    Create combined frame:
      Top row: [external MuJoCo view of robot]
      Bottom row: [desired | current | diff]
    """
    des_bgr = to_uint8_bgr(rgb_des)
    cur_bgr = to_uint8_bgr(rgb_cur)
    diff = torch.clamp(torch.abs(rgb_des - rgb_cur) * 3.0, 0, 1)
    diff_bgr = to_uint8_bgr(diff)

    H, W = des_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    s = 0.4

    cv2.putText(des_bgr, "Desired", (5, 15), font, s, (255, 255, 255), 1)
    cv2.putText(cur_bgr, "Current (UR10e)", (5, 15), font, s, (255, 255, 255), 1)
    cv2.putText(diff_bgr, "|Diff| x3", (5, 15), font, s, (255, 255, 255), 1)

    cv2.putText(cur_bgr, f"it={it} err={err:.0f}", (5, H - 35), font, s, (255, 255, 255), 1)
    cv2.putText(cur_bgr, f"t={t_err:.4f}m r={r_err:.1f}deg", (5, H - 20), font, s, (255, 255, 255), 1)
    cv2.putText(cur_bgr, f"{mode} sf={sf:.2f}", (5, H - 5), font, s, (0, 255, 255), 1)

    if q_deg is not None:
        cv2.putText(diff_bgr, f"UR10e q(deg):", (5, H - 35), font, 0.3, (200, 200, 200), 1)
        cv2.putText(diff_bgr, f" {q_deg[:3]}", (5, H - 20), font, 0.3, (200, 200, 200), 1)
        cv2.putText(diff_bgr, f" {q_deg[3:]}", (5, H - 5), font, 0.3, (200, 200, 200), 1)

    # Bottom row: camera views
    bottom = np.concatenate([des_bgr, cur_bgr, diff_bgr], axis=1)

    if mj_frame is not None:
        # Resize MuJoCo frame to match bottom row width
        total_w = bottom.shape[1]
        mj_h = int(mj_frame.shape[0] * total_w / mj_frame.shape[1])
        mj_resized = cv2.resize(
            cv2.cvtColor(mj_frame, cv2.COLOR_RGB2BGR),
            (total_w, mj_h))
        cv2.putText(mj_resized, "External View (MuJoCo)", (10, 25),
                    font, 0.6, (255, 255, 255), 1)
        cv2.putText(mj_resized, f"it={it}  {mode}  t={t_err:.3f}m  r={r_err:.1f}deg",
                    (10, mj_h - 10), font, 0.5, (0, 255, 255), 1)
        return np.concatenate([mj_resized, bottom], axis=0)

    return bottom


def run_ur_vs(mode, scale_factor, c2w_start, c2w_goal,
              means, quats, scales_original, opacities, colors,
              sh_degree, K_np, W, H, cam_params,
              camera_model, feature_type,
              mu, lambda_gain, max_iter, convergence_threshold,
              robot_model, save_every, out_dir,
              all_camtoworlds=None, device="cuda"):
    """Run VS with UR robot kinematics tracking."""
    os.makedirs(out_dir, exist_ok=True)

    sf = scale_factor if mode == "inflated" else 1.0
    scales_cur = scales_original * sf

    # Camera simulator (proven to work)
    wMo = np.eye(4)
    robot_cam = SimulatorCamera()
    cMo_start = np.linalg.inv(c2w_start)
    robot_cam.setPosition(wMo @ np.linalg.inv(cMo_start))
    robot_cam.setRobotState(1)
    cMo = cMo_start.copy()

    # UR robot: place base at scene center on the floor
    ur = UR_Simulator(model=robot_model, dt=0.04)

    # Compute base position from camera positions
    cam_positions = np.array([all_camtoworlds[i][:3, 3]
                              for i in range(len(all_camtoworlds))])
    base_pos = cam_positions.mean(axis=0).copy()
    base_pos[2] = cam_positions[:, 2].min() - 0.15  # below lowest camera (floor)
    ur.place_base_fixed(base_pos)

    # Move robot to start camera pose via IK
    print(f"[{mode}] Robot base at {base_pos}")
    ok, (ik_t, ik_r) = ur.move_to_pose(c2w_start, max_iter=5000)
    print(f"[{mode}] IK to start: {'OK' if ok else 'FAIL'} "
          f"(t={ik_t:.4f}m, r={ik_r:.2f}deg)")

    # MuJoCo renderer for external view with GS scene background
    mj_renderer = MuJoCoUR10eRenderer(width=640, height=480)
    mj_renderer.set_camera(distance=1.2, azimuth=170, elevation=-10,
                           lookat=(0.15, 0, 0.45))
    mj_renderer.set_eMc(ur.eMc)
    # Initialize GS scene for background rendering
    mj_renderer.init_gs_scene(
        means=means, quats=quats, scales=scales_original,
        opacities=opacities, colors=colors, sh_degree=sh_degree,
        bMo=ur.bMo, device=device)

    # Goal camera pose in base frame for frustum visualization
    # ur.bMo maps object→base (inv of oMb). c2w_goal maps camera→object.
    # bMc = bMo @ c2w_goal (object→base composed with camera→object)
    bMc_goal_pose = ur.bMo @ c2w_goal

    # Render desired
    cMo_goal = np.linalg.inv(c2w_goal)
    rgb_des, gray_des, depth_des = render_gsplat(
        cMo_goal, means, quats, scales_cur, opacities, colors,
        sh_degree, K_np, W, H, camera_model=camera_model, device=device)
    s_star = create_feature(feature_type, device=device, border=10)
    s_star.init(H, W)
    s_star.setCameraParameters(cam_params)
    s_star.buildFrom(gray_des, depth_des)

    errors, pose_t, pose_r = [], [], []
    joint_history = [ur.q.copy()]

    print(f"\n[{mode}] Running (sf={sf:.2f}, robot={robot_model})...")
    print(f"{'Iter':>6} {'Error':>10} {'d(m)':>8} {'r(deg)':>8}")
    print("-" * 40)

    for it in range(max_iter):
        # Render from current camera pose
        rgb_cur, gray_cur, depth_cur = render_gsplat(
            cMo, means, quats, scales_cur, opacities, colors,
            sh_degree, K_np, W, H, camera_model=camera_model, device=device)

        s = create_feature(feature_type, device=device, border=10)
        s.init(H, W)
        s.setCameraParameters(cam_params)
        s.buildFrom(gray_cur, depth_cur)

        error = s.error(s_star)
        err_norm = torch.sum(error ** 2).item()
        errors.append(err_norm)

        c2w_cur = np.linalg.inv(cMo)
        t_err, r_err = se3_distance(c2w_cur, c2w_goal)
        pose_t.append(t_err)
        pose_r.append(r_err)

        # Save frame with MuJoCo external view
        if it % save_every == 0:
            mj_frame = mj_renderer.render_composite(ur.q, bMc_goal=bMc_goal_pose)
            frame = make_frame(rgb_des, rgb_cur, it, err_norm, t_err, r_err,
                               ur.get_joints_deg(), mode, sf, mj_frame=mj_frame)
            cv2.imwrite(os.path.join(out_dir, f"frame_{it:05d}.png"), frame)
            print(f"{it:>6} {err_norm:>10.0f} {t_err:>8.4f} {r_err:>8.2f}")

        # Convergence
        if t_err < 0.005 and r_err < 0.5:
            mj_frame = mj_renderer.render_composite(ur.q, bMc_goal=bMc_goal_pose)
            frame = make_frame(rgb_des, rgb_cur, it, err_norm, t_err, r_err,
                               ur.get_joints_deg(), mode, sf, mj_frame=mj_frame)
            cv2.imwrite(os.path.join(out_dir, f"frame_{it:05d}.png"), frame)
            print(f"  CONVERGED at it={it}")
            break

        if sf < 1.1 and err_norm < convergence_threshold:
            mj_frame = mj_renderer.render_composite(ur.q, bMc_goal=bMc_goal_pose)
            frame = make_frame(rgb_des, rgb_cur, it, err_norm, t_err, r_err,
                               ur.get_joints_deg(), mode, sf, mj_frame=mj_frame)
            cv2.imwrite(os.path.join(out_dir, f"frame_{it:05d}.png"), frame)
            print(f"  CONVERGED at it={it}")
            break

        # VS velocity (LM)
        Ls = s.interaction()
        Hs = Ls.T @ Ls
        diagHs = torch.diag(torch.diag(Hs))
        Hess = torch.linalg.inv(
            mu * diagHs + Hs + 1e-6 * torch.eye(6, device=device))
        v_cam = -lambda_gain * (Hess @ Ls.T @ error)
        v_cam_np = v_cam.detach().cpu().numpy()

        vt = np.linalg.norm(v_cam_np[:3])
        vr = np.linalg.norm(v_cam_np[3:])
        if vt > 0.5: v_cam_np[:3] *= 0.5 / vt
        if vr > 0.3: v_cam_np[3:] *= 0.3 / vr

        # Apply velocity via SimulatorCamera (exponential map, proven)
        robot_cam.setVelocity("camera", v_cam_np)
        wMc = robot_cam.getPosition()
        cMo = np.linalg.inv(wMc) @ wMo

        # Track equivalent UR joint motion
        ur.set_camera_velocity(v_cam_np)
        joint_history.append(ur.q.copy())

    else:
        print(f"  FAILED after {max_iter} iterations")
        print(f"  Final: t={t_err:.4f}m, r={r_err:.2f}deg")

    n_frames = len([f for f in os.listdir(out_dir) if f.startswith("frame_")])
    print(f"  Saved {n_frames} frames to {out_dir}/")

    np.savez(os.path.join(out_dir, f"ur_results_{mode}.npz"),
             errors=np.array(errors), pose_t=np.array(pose_t),
             pose_r=np.array(pose_r), joints=np.array(joint_history),
             mode=mode, scale_factor=sf)


def main():
    p = argparse.ArgumentParser(
        description="UR10e PVS with 3D Gaussian Splatting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--data_factor", type=int, default=8)
    p.add_argument("--camera_model", default="pinhole")
    p.add_argument("--feature_type", default="pinhole")
    p.add_argument("--robot_model", default="ur10e", choices=["ur5e", "ur10e"])

    p.add_argument("--goal_idx", type=int, default=10)
    p.add_argument("--start_idx", type=int, default=12)
    p.add_argument("--modes", nargs="+", default=["original", "inflated"])
    p.add_argument("--scale_factor", type=float, default=1.8)

    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--lambda_", type=float, default=10.0)
    p.add_argument("--convergence_threshold", type=float, default=10000)
    p.add_argument("--max_iter", type=int, default=500)

    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--out_dir", type=str, default="logs/ur10e_sim")
    args = p.parse_args()
    device = "cuda"

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
    convergence_threshold = args.convergence_threshold * pixel_ratio

    c2w_goal = camtoworlds[args.goal_idx]
    c2w_start = camtoworlds[args.start_idx]
    init_t, init_r = se3_distance(c2w_start, c2w_goal)

    print(f"[Scene] {len(camtoworlds)} views, {W}x{H} (factor={df})")
    print(f"[Robot] {args.robot_model.upper()}")
    print(f"[Pair]  view {args.start_idx} -> {args.goal_idx}: "
          f"d={init_t:.3f}m, r={init_r:.1f}deg")

    for mode in args.modes:
        run_ur_vs(
            mode=mode, scale_factor=args.scale_factor,
            c2w_start=c2w_start, c2w_goal=c2w_goal,
            means=means, quats=quats, scales_original=scales_original,
            opacities=opacities, colors=colors,
            sh_degree=sh_degree, K_np=K_np, W=W, H=H,
            cam_params=cam_params, camera_model=args.camera_model,
            feature_type=args.feature_type,
            mu=args.mu, lambda_gain=args.lambda_,
            max_iter=args.max_iter,
            convergence_threshold=convergence_threshold,
            robot_model=args.robot_model,
            save_every=args.save_every,
            out_dir=os.path.join(args.out_dir, mode),
            all_camtoworlds=camtoworlds,
            device=device,
        )

    print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
