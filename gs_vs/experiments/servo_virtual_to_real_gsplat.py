import argparse
import time
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from simulators.RobotSimulator import SimulatorCamera
from simulators.CameraParameters import CameraParameters

#from gs_vs.FeatureLuminancePinhole import FeatureLuminancePinhole as FeatureLuminanceTorch
from gs_vs.FeatureLuminanceUnifiedCS import FeatureLuminanceUnifiedCS as FeatureLuminanceTorch
#from gs_vs.FeatureLuminanceUnifiedPS import FeatureLuminanceUnifiedPS as FeatureLuminanceTorch
#from gs_vs.FeatureLuminanceUnifiedIP import FeatureLuminanceUnifiedIP as FeatureLuminanceTorch

from gs_vs.tools.image_tools import save_rendered_images, compute_fisheye_mask, normalize_mad
from gs_vs.tools.SE3_tools import exponential_map


from gs_vs.datasets.colmap import Parser
from gs_vs.gsplat.rendering import rasterization

from gs_vs.viewers.viewer_viser import VsViserViewer
from PIL import Image
import torchvision.transforms as T
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
    """
    Render using gsplat
    
    Args:
        cMo: Camera pose
        means, quats, scales, opacities, colors: Gaussian splat parameters
        sh_degree: Spherical harmonics degree
        K_np: Intrinsic matrix
        W, H: Image dimensions
        camera_model: "pinhole" or "fisheye" (gsplat terminology)
        device: torch device
        
    Returns:
        rgb, gray, depth, mask: Rendered outputs
    """
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
    camtoworld = camtoworlds[0].cpu().numpy()
    cdMo = np.linalg.inv(camtoworld)

    K_np = list(parser.Ks_dict.values())[0]
    W, H = list(parser.imsize_dict.values())[0]
    
    W, H = 640, 512

    # === Camera-specific setup ===

    
    if args.camera_model == 'fisheye':
        
        # for 1024x1024
        #fpx = 633.446
        #fpy = 633.775
        #u0 = 503.83
        #v0 = 492.238       
        
        # Fujinon fisheye parameters
        fpx = 316.723    # 633.446 / 2
        fpy = 316.888    # 633.775 / 2
        u0  = 315.915    # (503.83 + 128) / 2
        v0  = 246.119    # 492.238 / 2

        # Intrinsic matrix
        K_np = np.array([
            [fpx, 0.0,   u0],
            [0.0,  fpy,  v0],
            [0.0,  0.0,   1.0]
        ])
        
        # Create circular mask for fisheye
        r_max = max(u0, v0)# / 2.0
        mask = compute_fisheye_mask(W, H, u0, v0, r_max)
        print(f"Created fisheye mask with radius {r_max:.1f} pixels")
        
    elif args.camera_model =='pinhole':
        mask = torch.ones((H, W), dtype=torch.bool, device=device)
        pass
        

    # === Desired image (loaded from disk) ===
    print(f"Loading desired image from: {args.desired_image}")

    # Load image
    pil_img = Image.open(args.desired_image).convert("RGB")

    # Resize if necessary
    if pil_img.size != (W, H):
        print(f"Resizing desired image from {pil_img.size} to {(W, H)}")
        pil_img = pil_img.resize((W, H), Image.BILINEAR)

    # Convert to tensor [H,W,3] in [0,1]
    rgb_des = T.ToTensor()(pil_img).permute(1, 2, 0).to(device)

    # Convert to grayscale [0,255]
    gray_des = (
        0.2989 * rgb_des[..., 0] +
        0.5870 * rgb_des[..., 1] +
        0.1140 * rgb_des[..., 2]
    ) * 255.0
    
    gray_des = normalize_mad(gray_des)
    
    # Since we don't have depth for real image
    depth_des = torch.ones((H, W), device=device)

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

    # Desired features (s_star)
    s_star = FeatureLuminanceTorch(
        device="cuda",
        border=10
    )
    s_star.init(H, W)
    s_star.setCameraParameters(cam_params)
    s_star.buildFrom(gray_des, depth_des, mask=mask)

    # Current features (s)
    s = FeatureLuminanceTorch(
        device="cuda",
        border=10
    )
    s.init(H, W)
    s.setCameraParameters(cam_params)

    # === Initial pose ===
    cMo = cdMo.copy()
    cMo[:3, 3] += [2.0, 0.0, 0.5]
    #cMo[:3, 3] += [2.0, -0.2, -0.5]  # Initial displacement
    #cMo[:3, 3] += [1.2, 0.2, -0.2]  # Initial displacement
    #cMo[:3, 3] += [0.1, 0.05, -0.05]  # Plus petit

    # === Initial image ===
    print("Rendering initial image...")
    rgb_ini, gray_ini, depth_ini, _ = render_gsplat(
        cMo, means, quats, scales, opacities, colors,
        sh_degree, K_np, W, H,
        camera_model=args.camera_model
    )
    gray_ini = normalize_mad(gray_ini)
    print('min max mini', gray_ini.min().item(), gray_ini.max().item())
    
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
        gray = normalize_mad(gray)
            
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
        rgb_masked[~mask] = 0
            
        viewer.update(
            iteration=it,
            cMo=np.linalg.inv(cMo),
            rgb=rgb_masked,
            error=err_norm,
            velocity=v_np,
        )

        #if it % 10 == 0:
        #    print(f"Iter {it:4d}: error = {err_norm:10.2f}, |v| = {np.linalg.norm(v_np):.4f}")

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
    
    parser.add_argument("--desired_image",
        required=True,
        help="Path to desired image (RGB)"
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
