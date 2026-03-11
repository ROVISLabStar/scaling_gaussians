import argparse
import time
import torch
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

from gs_vs.simulators.RobotSimulator import SimulatorCamera
from gs_vs.simulators.CameraParameters import CameraParameters

from gs_vs.features.factory import create_feature

from gs_vs.tools.image_tools import save_rendered_images, compute_fisheye_mask_v2
from gs_vs.tools.SE3_tools import exponential_map
from gs_vs.viewers.viewer_viser import VsViserViewer


from gs_vs.datasets.colmap import Parser
from gsplat.rendering import rasterization



# ==============================
# Parameters
# ==============================
mu = 0.01
lambda_ = 10.0
max_iter = 4000

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
    
    packed = True
    
    renders, render_alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        sh_degree=sh_degree,
        viewmats=viewmat,
        Ks=Ks,
        width=W,
        height=H,
        packed=packed,
        render_mode="RGB+ED",
        camera_model=camera_model,
    )
    
    rgb   = renders[0, ..., :3]
    depth = renders[0, ..., 3]
    rgb = torch.clamp(rgb, 0.0, 1.0)

    # Convert to grayscale
    gray = (
        0.2989 * rgb[..., 0] +
        0.5870 * rgb[..., 1] +
        0.1140 * rgb[..., 2]
    ) * 255.0

    mask = (render_alphas[0, ..., 0] > 1e-4)

    return rgb, gray, depth, mask

def load_basic_cfg_fields(cfg_path):
    """Load configuration from file"""
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
    



# ==============================
# Main
# ==============================
@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load config ----
    cfg = load_basic_cfg_fields(args.cfg)
    parser = Parser(
        data_dir=cfg["data_dir"],
        factor=cfg["data_factor"],
        normalize=cfg["normalize_world_space"],
        test_every=8,
    )
    
    # === Load checkpoint ===
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    splats = ckpt["splats"]
    means = splats["means"].to(device)
    quats = splats["quats"].to(device)
    scales = torch.exp(splats["scales"]).to(device)
    opacities = torch.sigmoid(splats["opacities"]).to(device)
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1).to(device)
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)

    camtoworlds = torch.from_numpy(parser.camtoworlds).float().to(device)
    
    intrinsics_file = args.intrinsics_file

    if intrinsics_file is not None:
        print(f"[INFO] Loading intrinsics from file: {intrinsics_file}")

        with open(intrinsics_file, "r") as f:
            intr = yaml.safe_load(f)

        W = intr["width"]
        H = intr["height"]
        fx = intr["fx"]
        fy = intr["fy"]
        cx = intr["cx"]
        cy = intr["cy"]

    else:
    
        print("[INFO] Using intrinsics from COLMAP parser")

        W, H = list(parser.imsize_dict.values())[0]
        K_colmap = torch.from_numpy(
            list(parser.Ks_dict.values())[0]
        ).float()

        fx = K_colmap[0, 0].item()
        fy = K_colmap[1, 1].item()
        cx = K_colmap[0, 2].item()
        cy = K_colmap[1, 2].item()
        
    K_np = np.array([
        [fx, 0.0,   cx],
        [0.0,  fy,  cy],
        [0.0,  0.0,   1.0]
    ])
        
    apply_mask = (args.camera_model == "fisheye")

    mask = None
    if apply_mask:
        print("[INFO] Fisheye mode → mask enabled")
        mask = compute_fisheye_mask_v2(W, H, cx, cy)
            
    idx = int(args.desired_image_index)
    print(f"[INFO] getting view index {idx}")
    camtoworld = camtoworlds[idx].cpu().numpy()
    cdMo = np.linalg.inv(camtoworld)

 
    
    rgb_des, gray_des, depth_des, _ = render_gsplat(
        cdMo, means, quats, scales, opacities, colors,
        sh_degree, K_np, W, H,
        camera_model=args.camera_model
    )

    if apply_mask:
        # Apply mask
        rgb_des[~mask] = 0
        gray_des[~mask] = 0
        depth_des[~mask] = 0
    
    save_rendered_images(rgb_des, gray_des, depth_des, mask=mask, 
                        out_dir="logs/comp", prefix="des")

    # === Feature setup ===
    cam_params = CameraParameters(
        px=K_np[0, 0], py=K_np[1, 1],
        u0=K_np[0, 2], v0=K_np[1, 2],
    )

    feature_type = args.feature_type  # new CLI argument

    print(f"[INFO] feature: {feature_type}")



    # Desired features (s_star)
    s_star = create_feature(
        feature_type,
        device="cuda",
        border=10
    )
    s_star.init(H, W)
    s_star.setCameraParameters(cam_params)
    s_star.buildFrom(gray_des, depth_des, mask=mask)

    # Current features (s)
    s = create_feature(
        feature_type,
        device="cuda",
        border=10
    )

    s.init(H, W)
    s.setCameraParameters(cam_params)

    # === Initial pose ===
    cMo = cdMo.copy()
    #cMo[:3, 3] += [2.0, -0.2, -0.5]  # Initial displacement
    cMo[:3, 3] += [0.2, 0.2, -0.2]  # Initial displacement
    #cMo[:3, 3] += [2.0, 0.05, -0.05]  # Plus petit

    # === Initial image ===
    print("Rendering initial image...")
    rgb_ini, gray_ini, depth_ini, _ = render_gsplat(
        cMo, means, quats, scales, opacities, colors,
        sh_degree, K_np, W, H,
        camera_model=args.camera_model
    )
    if apply_mask:
        rgb_ini[~mask] = 0
        gray_ini[~mask] = 0
        depth_ini[~mask] = 0
        
    save_rendered_images(rgb_ini, gray_ini, depth_ini, mask=mask, 
                        out_dir="logs/comp", prefix="ini")

    # Initialize robot simulator
    robot = SimulatorCamera()
    wMo = np.eye(4)
    robot.setPosition(wMo @ np.linalg.inv(cMo))
    robot.setRobotState(1)

    # === Initialize Viewer ===
    viewer = VsViserViewer(
        rgb_des=rgb_des.cpu().numpy(),
        cdMo=np.linalg.inv(cdMo),
        rgb_cur=rgb_ini.cpu().numpy(),
        cMo=np.linalg.inv(cMo),
        image_size=(W, H),
        aspect_ratio=W/H,
        server_port=8080,
        image_scale=0.5,
        means=means.detach().cpu().numpy(),
        scales=scales.detach().cpu().numpy(),
        quats=quats.detach().cpu().numpy(),
        sh0=splats["sh0"].detach().cpu().numpy(),
        opacities=opacities.detach().cpu().numpy()
    )

    # === Visual Servo Loop ===
    print(f"\nStarting visual servoing (max {max_iter} iterations)...")
    print(f"Using μ={mu}, λ={lambda_}")
    
    for it in range(max_iter):
        # Render current image
        rgb, gray, depth, _ = render_gsplat(
            cMo, means, quats, scales, opacities, colors,
            sh_degree, K_np, W, H,
            camera_model=args.camera_model
        )
        if apply_mask:
            # Apply mask
            rgb[~mask] = 0
            gray[~mask] = 0
            depth[~mask] = 0

        # Build current features with mask
        s.buildFrom(gray, depth, mask=mask)
        
        # Compute error (only masked pixels)
        error = s.error(s_star)
        err_norm = torch.sum(error ** 2).item()
        
        # Compute interaction matrix (only masked pixels contribute)
        Ls = s.interaction()
        #print('Ls', Ls)
        Hs = Ls.T @ Ls
        diagHs = torch.diag(torch.diag(Hs))
        Hess = torch.linalg.inv(mu * diagHs + Hs + 1e-6 * torch.eye(6, device=device))

        # Compute velocity
        v = -lambda_ * (Hess @ Ls.T @ error)

        # Update robot
        v_np = v.detach().cpu().numpy()
        robot.setVelocity("camera", v_np)
        wMc = robot.getPosition()
        cMo = np.linalg.inv(wMc) @ wMo

        s.reset()

        # === Viewer update ===
        rgb_masked = rgb.clone()
        if apply_mask:
            rgb_masked[~mask] = 0
            
        viewer.update(
            iteration=it,
            cMo=np.linalg.inv(cMo),
            rgb=rgb_masked,
            error=err_norm,
            velocity=v_np,
        )

        time.sleep(0.1)
        
        # Check convergence
        if err_norm < 1e2:
            print(f"\nConvergence achieved at iteration {it}!")
            break
    
    if it == max_iter - 1:
        print(f"\nReached maximum iterations ({max_iter})")
    
    # Save recording
    viewer.save("logs/servo_recording.viser")
    print("Servo recording saved to logs/servo_recording.viser")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Photometric Visual Servoing with Gaussian Splatting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--ckpt", required=True, 
                       help="Path to checkpoint file")
    parser.add_argument("--cfg", required=True,
                       help="Path to configuration file")
    
    # Camera model (gsplat terminology)
    parser.add_argument("--camera_model", default="pinhole", 
                       choices=["pinhole", "fisheye", "spherical", "ortho"],
                       help="Camera model for rendering (gsplat terminology)")
                       
    parser.add_argument("--intrinsics_file", default=None)
    parser.add_argument("--desired_image_index", default=0)
    
    parser.add_argument(
        "--feature_type",
        default="unified_ps",
        choices=["pinhole", "unified_ip", "unified_cs", "unified_ps", "equidistant"],
    )

    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*60)
    print("PHOTOMETRIC VISUAL SERVOING WITH GAUSSIAN SPLATTING")
    print("="*60)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Config: {args.cfg}")
    print(f"Gsplat Camera Model: {args.camera_model}")
    print("="*60 + "\n")
    
    main(args)
