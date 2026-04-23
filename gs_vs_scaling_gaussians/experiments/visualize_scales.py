"""
Visualize 3DGS Rendering at Different Gaussian Scales
======================================================

Multiplies ALL Gaussian scale parameters by a factor and renders
the same view. Larger scales → more blur → smoother cost landscape
→ larger convergence basin.

This demonstrates the direct link between 3DGS Gaussian scales
and the blur parameter in Naamani/Caron IROS 2024.

Usage:
    python experiments/visualize_scales.py \
        --ckpt ... --cfg ... --view_idx 128

Author: Youssef (UM6P / Ai Movement Lab)
"""

import argparse
import os
import torch
import numpy as np

from gs_vs.datasets.colmap import Parser
from gsplat.rendering import rasterization


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


def save_rgb(tensor, path):
    import cv2
    img = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--view_idx", type=int, default=128)
    p.add_argument("--data_factor", type=int, default=4,
                   help="Downscale for faster rendering")
    p.add_argument("--camera_model", default="pinhole")
    p.add_argument("--out_dir", type=str, default="logs/scale_viz")
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
    scales_log = splats["scales"].to(device)  # log-space scales
    opacities = torch.sigmoid(splats["opacities"]).to(device)
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1).to(device)
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)

    # Original scales
    scales_original = torch.exp(scales_log)

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

    c2w = camtoworlds[args.view_idx]
    cMo = np.linalg.inv(c2w)

    print(f"[Scene] {len(camtoworlds)} views, render at {W}x{H}")
    print(f"[View] {args.view_idx}")
    print(f"[Gaussians] {len(means)} primitives")
    print(f"[Scales] original: mean={scales_original.mean().item():.6f}, "
          f"std={scales_original.std().item():.6f}, "
          f"min={scales_original.min().item():.6f}, "
          f"max={scales_original.max().item():.6f}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Render at different scale multipliers
    scale_factors = [0.5, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]

    viewmat = torch.from_numpy(cMo).float().to(device)[None]
    Ks = torch.from_numpy(K_np).float().to(device)[None]

    print(f"\n{'Factor':<10} {'Mean scale':<15} {'Projected σ (px)':<18} {'Image'}")
    print("-" * 65)

    for sf in scale_factors:
        # Multiply scales
        scales_modified = scales_original * sf

        with torch.no_grad():
            renders, _, _ = rasterization(
                means=means, quats=quats, scales=scales_modified,
                opacities=opacities, colors=colors,
                sh_degree=sh_degree, viewmats=viewmat, Ks=Ks,
                width=W, height=H, packed=True,
                render_mode="RGB+ED", camera_model=args.camera_model,
            )
            rgb = torch.clamp(renders[0, ..., :3], 0.0, 1.0)
            depth = renders[0, ..., 3]

        # Estimate mean projected Gaussian size in pixels
        # σ_2D ≈ fx * scale_3D / Z
        valid_depth = depth[depth > 0]
        if len(valid_depth) > 0:
            mean_Z = valid_depth.mean().item()
            mean_sigma_px = fx * scales_modified.mean().item() / mean_Z
        else:
            mean_sigma_px = 0

        fname = f"scale_{sf:.2f}.png"
        save_rgb(rgb, os.path.join(args.out_dir, fname))

        print(f"{sf:<10.2f} {scales_modified.mean().item():<15.6f} "
              f"{mean_sigma_px:<18.2f} {fname}")

    # Also render with Gaussian image blur for comparison
    # (blur the original rendering with different kernel sizes)
    import cv2

    # Render original
    with torch.no_grad():
        renders, _, _ = rasterization(
            means=means, quats=quats, scales=scales_original,
            opacities=opacities, colors=colors,
            sh_degree=sh_degree, viewmats=viewmat, Ks=Ks,
            width=W, height=H, packed=True,
            render_mode="RGB+ED", camera_model=args.camera_model,
        )
        rgb_original = torch.clamp(renders[0, ..., :3], 0.0, 1.0)

    rgb_np = (rgb_original.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    print(f"\nImage-space Gaussian blur comparison:")
    print(f"{'σ_blur (px)':<15} {'Image'}")
    print("-" * 30)

    for sigma in [0, 1, 2, 5, 10, 20, 50]:
        if sigma == 0:
            blurred = rgb_np
        else:
            ksize = int(6 * sigma + 1) | 1  # ensure odd
            blurred = cv2.GaussianBlur(rgb_np, (ksize, ksize), sigma)

        fname = f"blur_sigma_{sigma:02d}.png"
        cv2.imwrite(os.path.join(args.out_dir, fname),
                    cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))
        print(f"{sigma:<15} {fname}")

    print(f"\n[Saved] All images to {args.out_dir}/")
    print(f"\nCompare:")
    print(f"  scale_*.png  — 3DGS with modified Gaussian scales (structural blur)")
    print(f"  blur_*.png   — Image-space Gaussian blur (post-processing blur)")
    print(f"\nThe key insight: 3DGS scale modification is NOT the same as image blur.")
    print(f"Scale changes affect the 3D structure (Gaussians become larger/smaller),")
    print(f"while image blur is a 2D convolution. Both expand the convergence basin")
    print(f"but through different mechanisms.")


if __name__ == "__main__":
    main()
