import argparse
import os
import yaml
import torch
import imageio
import numpy as np
from tqdm import tqdm

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from datasets.colmap import Parser
from gsplat.rendering import rasterization

import sys
sys.path.append("../../")

from tools.image_tools import save_rendered_images, compute_fisheye_mask_v2

@torch.no_grad()
def main(args):
    device = torch.device("cuda:0")

    # ---------------------------------------------------------
    # 1. Load cfg.yml as raw metadata (NO Config reconstruction)
    # ---------------------------------------------------------
    with open(args.cfg, "r") as f:
        cfg_raw = yaml.load(f, Loader=yaml.Loader)

    data_dir = cfg_raw["data_dir"]
    data_factor = cfg_raw["data_factor"]
    normalize_world_space = cfg_raw["normalize_world_space"]
    test_every = cfg_raw["test_every"]

    camera_model = args.camera_model or cfg_raw["camera_model"]
    near_plane = cfg_raw["near_plane"]
    far_plane = cfg_raw["far_plane"]

    print(f"[INFO] camera_model = {camera_model}")

    # ---------------------------------------------------------
    # 2. Load training cameras (EXACTLY like training)
    # ---------------------------------------------------------
    parser = Parser(
        data_dir=data_dir,
        factor=data_factor,
        normalize=normalize_world_space,
        test_every=test_every,
    )

    camtoworlds = torch.from_numpy(parser.camtoworlds).float().to(device)
    
    # ---------------------------------------------------------
    # Override intrinsics with custom focal
    # ---------------------------------------------------------
    intrinsics_file = args.intrinsics_file

    if intrinsics_file is not None:
        print(f"[INFO] Loading intrinsics from file: {intrinsics_file}")

        with open(intrinsics_file, "r") as f:
            intr = yaml.safe_load(f)

        width = intr["width"]
        height = intr["height"]
        fx = intr["fx"]
        fy = intr["fy"]
        cx = intr["cx"]
        cy = intr["cy"]

    else:
        print("[INFO] Using intrinsics from COLMAP parser")

        width, height = list(parser.imsize_dict.values())[0]
        K_colmap = torch.from_numpy(
            list(parser.Ks_dict.values())[0]
        ).float()

        fx = K_colmap[0, 0].item()
        fy = K_colmap[1, 1].item()
        cx = K_colmap[0, 2].item()
        cy = K_colmap[1, 2].item()

    # Build intrinsic matrix
    K_single = torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    
    # ---------------------------------------------------------
    # 3. Load Gaussian splats from checkpoint
    # ---------------------------------------------------------
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    splats = ckpt["splats"]

    means = splats["means"].to(device)
    quats = splats["quats"].to(device)
    scales = torch.exp(splats["scales"]).to(device)
    opacities = torch.sigmoid(splats["opacities"]).to(device)

    sh0 = splats["sh0"].to(device)
    shN = splats["shN"].to(device)
    colors = torch.cat([sh0, shN], dim=1)

    sh_degree = int(np.sqrt(colors.shape[1]) - 1)

    print(f"[INFO] Loaded {len(means)} Gaussians")
    print(f"[INFO] SH degree = {sh_degree}")

    # ---------------------------------------------------------
    # 4. Render training views
    # ---------------------------------------------------------
    render_dir = args.result_dir
    os.makedirs(render_dir, exist_ok=True)
        
    apply_mask = (camera_model == "fisheye")
    compute_metrics = (camera_model == "pinhole")

    if apply_mask:
        print("[INFO] Fisheye mode → mask enabled")
        mask = compute_fisheye_mask_v2(width, height, cx, cy)
        mask = mask.detach().cpu().numpy()
    if compute_metrics:
        print("[INFO] Pinhole mode → computing PSNR & SSIM")

        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        psnr_values = []
        ssim_values = []
    
    for i in tqdm(range(len(camtoworlds)), desc="Rendering training views"):
        c2w = camtoworlds[i : i + 1]      # [1, 4, 4]
        K = K_single[None]               # [1, 3, 3]

        render_colors, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(c2w),
            Ks=K,
            width=width,
            height=height,
            sh_degree=sh_degree,
            camera_model=camera_model,
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode="RGB",
        )
        rgb_torch = torch.clamp(render_colors[0], 0.0, 1.0)
        rgb = rgb_torch.cpu().numpy()

        if compute_metrics:
            gt_path = parser.image_paths[i]
            gt_np = imageio.imread(gt_path) / 255.0
            gt = torch.from_numpy(gt_np).float().to(device)

            # Ensure resolution match
            if gt.shape[0] != height or gt.shape[1] != width:
                raise ValueError("GT resolution mismatch")

            # Convert to [B,C,H,W]
            rgb_t = rgb_torch.permute(2, 0, 1).unsqueeze(0)
            gt_t  = gt.permute(2, 0, 1).unsqueeze(0)

            psnr_val = psnr_metric(rgb_t, gt_t)
            ssim_val = ssim_metric(rgb_t, gt_t)

            psnr_values.append(psnr_val.item())
            ssim_values.append(ssim_val.item())
        
        if apply_mask:
            rgb[~mask] = 0.0
        
        imageio.imwrite(
            os.path.join(render_dir, f"train_view_{i:04d}.png"),
            (rgb * 255).astype(np.uint8),
        )
  
    if compute_metrics:
        mean_psnr = np.mean(psnr_values)
        mean_ssim = np.mean(ssim_values)

        print("\n[RESULTS - PINHOLE]")
        print(f"Mean PSNR: {mean_psnr:.4f}")
        print(f"Mean SSIM: {mean_ssim:.4f}")
        
    print(f"[DONE] Images saved to {render_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--result_dir", default="train_renders")
    parser.add_argument(
        "--camera_model",
        choices=["pinhole", "ortho", "fisheye"],
        default="fisheye",
    )
    parser.add_argument("--intrinsics_file", default=None)

    
    args = parser.parse_args()
    main(args)

