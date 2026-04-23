"""
Interactive Scale Visualizer for 3DGS
======================================

Shows the effect of scaling α on:
- 3D Gaussian splats (in the 3D viewer)
- Rendered image (on a camera frustum)

Drag the α slider to see the Gaussians grow and the image blur in real-time.

Usage:
    python -m gs_vs_scaling_gaussians.viser.scale_visualizer \
        --ckpt <checkpoint> --cfg <config> --port 8080

Author: Youssef ALJ (UM6P)
"""

import argparse
import time
import numpy as np
import torch
import viser
import viser.transforms as tf

from gs_vs.simulators.CameraParameters import CameraParameters
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
            elif k == "normalize_world_space":
                data["normalize_world_space"] = v.lower() == "true"
    data.setdefault("data_factor", 1)
    data.setdefault("normalize_world_space", True)
    return data


def main():
    p = argparse.ArgumentParser(description="Interactive Scale Visualizer")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--data_factor", type=int, default=4)
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    device = "cuda"

    # Load scene
    cfg = load_basic_cfg_fields(args.cfg)
    parser = Parser(data_dir=cfg["data_dir"], factor=cfg["data_factor"],
                    normalize=cfg["normalize_world_space"], test_every=8)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    splats = ckpt["splats"]
    means_t = splats["means"].to(device)
    quats_t = splats["quats"].to(device)
    scales_original = torch.exp(splats["scales"]).to(device)
    opacities_t = torch.sigmoid(splats["opacities"]).to(device)
    colors_t = torch.cat([splats["sh0"], splats["shN"]], dim=1).to(device)
    sh_degree = int(np.sqrt(colors_t.shape[1]) - 1)

    camtoworlds = parser.camtoworlds
    n_views = len(camtoworlds)
    W_full, H_full = list(parser.imsize_dict.values())[0]
    K_colmap = list(parser.Ks_dict.values())[0]

    df = args.data_factor
    W, H = W_full // df, H_full // df
    fx, fy = K_colmap[0, 0] / df, K_colmap[1, 1] / df
    cx, cy = K_colmap[0, 2] / df, K_colmap[1, 2] / df
    K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    # Numpy data for splats
    means_np = splats["means"].cpu().numpy()
    quats_np = splats["quats"].cpu().numpy()
    scales_np = torch.exp(splats["scales"]).cpu().numpy()
    opacities_np = torch.sigmoid(splats["opacities"]).cpu().numpy()
    sh0_np = splats["sh0"].cpu().numpy()

    SH_C0 = 0.28209479177387814
    rgbs_np = np.clip(0.5 + SH_C0 * sh0_np[:, 0, :], 0.0, 1.0)

    print(f"[Scene] {n_views} views, {W}x{H}, {means_np.shape[0]} Gaussians")

    # Viser server
    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_layout="collapsible", control_width="large")
    print(f"[Viser] http://localhost:{args.port}")

    # Set up direction
    up_world = np.linalg.inv(camtoworlds[0])[:3, :3] @ np.array([0.0, -1.0, 0.0])
    up_world /= np.linalg.norm(up_world)
    server.scene.set_up_direction(tuple(up_world))

    # GUI
    with server.gui.add_folder("Scale Control"):
        alpha_slider = server.gui.add_slider(
            "Scale Factor (α)", min=0.5, max=5.0, step=0.1, initial_value=1.0)
        view_slider = server.gui.add_slider(
            "View Index", min=0, max=n_views - 1, step=1, initial_value=0)
        update_button = server.gui.add_button("Update")
        auto_update = server.gui.add_checkbox("Auto-update on slider change", True)
        alpha_text = server.gui.add_text("Current α", initial_value="1.0")

    # Training cameras (sparse)
    for i in range(0, n_views, max(1, n_views // 20)):
        wxyz = tf.SO3.from_matrix(camtoworlds[i][:3, :3]).wxyz
        server.scene.add_camera_frustum(
            f"/cameras/view_{i}", fov=np.pi/3, aspect=W/H, scale=0.02,
            wxyz=wxyz, position=camtoworlds[i][:3, 3], color=(128, 128, 128))
        server.scene.add_label(f"/cameras/label_{i}", text=str(i),
                               position=camtoworlds[i][:3, 3] + np.array([0, 0, 0.02]))

    # State
    splats_handle = [None]
    frustum_handle = [None]
    fov_rad = 2 * np.arctan(W / (2 * fx))

    def compute_covariances(alpha):
        """Compute scaled covariances for Gaussian splat display."""
        scaled_scales = scales_np * alpha
        Rs = tf.SO3(quats_np).as_matrix()
        covs = np.einsum(
            "nij,njk,nlk->nil", Rs,
            np.eye(3)[None, :, :] * scaled_scales[:, None, :] ** 2, Rs)
        return covs

    @torch.no_grad()
    def render_view(view_idx, alpha):
        """Render a view at given scale factor."""
        c2w = camtoworlds[view_idx]
        cMo = np.linalg.inv(c2w)
        scales_cur = scales_original * alpha
        viewmat = torch.from_numpy(cMo).float().to(device)[None]
        Ks = torch.from_numpy(K_np).float().to(device)[None]
        renders, _, _ = rasterization(
            means=means_t, quats=quats_t, scales=scales_cur,
            opacities=opacities_t, colors=colors_t,
            sh_degree=sh_degree, viewmats=viewmat, Ks=Ks,
            width=W, height=H, packed=True,
            render_mode="RGB+ED", camera_model="pinhole")
        rgb = torch.clamp(renders[0, ..., :3], 0.0, 1.0)
        return rgb.cpu().numpy()

    def update_scene():
        alpha = alpha_slider.value
        view_idx = view_slider.value
        alpha_text.value = f"{alpha:.1f}"

        # Update 3D Gaussians
        if splats_handle[0] is not None:
            splats_handle[0].remove()
        covs = compute_covariances(alpha)
        splats_handle[0] = server.scene.add_gaussian_splats(
            "/splats",
            centers=means_np,
            rgbs=rgbs_np,
            opacities=opacities_np[:, None],
            covariances=covs,
        )

        # Update rendered image on frustum
        img = render_view(view_idx, alpha)
        c2w = camtoworlds[view_idx]
        if frustum_handle[0] is not None:
            frustum_handle[0].remove()
        frustum_handle[0] = server.scene.add_camera_frustum(
            "/render_frustum", fov=fov_rad, aspect=W/H, scale=0.15,
            image=img[::2, ::2],  # downsample for display
            wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3], color=(0, 200, 0))

    # Initial render
    update_scene()

    @update_button.on_click
    def _(_):
        update_scene()

    @alpha_slider.on_update
    def _(_):
        if auto_update.value:
            update_scene()

    @view_slider.on_update
    def _(_):
        if auto_update.value:
            update_scene()

    print("[Viewer] Ready. Drag the α slider to see the effect.")
    print("  - 3D Gaussians resize in real-time")
    print("  - Rendered image updates on the frustum")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
