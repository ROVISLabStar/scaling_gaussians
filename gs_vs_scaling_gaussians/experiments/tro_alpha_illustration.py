"""
Didactic illustration of α effect on 3D Gaussians and their splatting.
=====================================================================

Shows 3 colored 3D ellipsoids rendered with different scale factors α,
demonstrating how inflating Gaussian scales smooths the rendered image.

Based on the pure-Python 3DGS renderer from:
https://colab.research.google.com/drive/1qoclD7HJ3-o0O1R8cvV3PxLhoDCMsH8W

Usage:
    python -m gs_vs_scaling_gaussians.experiments.tro_alpha_illustration

Author: Youssef ALJ (UM6P)
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import os


# ── 3DGS rendering (pure PyTorch, from 2DGS colab) ──

def build_rotation(r):
    norm = torch.sqrt((r * r).sum(dim=-1, keepdim=True))
    q = r / norm
    R = torch.zeros((q.size(0), 3, 3), device=q.device)
    r0, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r0*z)
    R[:, 0, 2] = 2 * (x*z + r0*y)
    R[:, 1, 0] = 2 * (x*y + r0*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r0*x)
    R[:, 2, 0] = 2 * (x*z - r0*y)
    R[:, 2, 1] = 2 * (y*z + r0*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_cov3d(scales, quats):
    R = build_rotation(quats)
    S = torch.zeros(scales.shape[0], 3, 3, device=scales.device)
    S[:, 0, 0] = scales[:, 0]
    S[:, 1, 1] = scales[:, 1]
    S[:, 2, 2] = scales[:, 2]
    L = R @ S
    return L @ L.transpose(1, 2)


def project_cov2d(means3D, cov3d, viewmat, fx, fy):
    t = (means3D @ viewmat[:3, :3]) + viewmat[-1:, :3]
    tz = t[..., 2]
    tx = t[..., 0]
    ty = t[..., 1]
    J = torch.zeros(means3D.shape[0], 3, 3, device=means3D.device)
    J[:, 0, 0] = fx / tz
    J[:, 0, 2] = -tx / (tz * tz) * fx
    J[:, 1, 1] = fy / tz
    J[:, 1, 2] = -ty / (tz * tz) * fy
    W = viewmat[:3, :3].T
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0, 2, 1)
    return cov2d[:, :2, :2], t


def render_gaussians(means3D, scales, quats, colors, opacities,
                     viewmat, fx, fy, cx, cy, W, H):
    """Simple differentiable 3DGS renderer (pure PyTorch)."""
    cov3d = build_cov3d(scales, quats)
    cov2d, t_view = project_cov2d(means3D, cov3d, viewmat, fx, fy)

    # Project means to 2D
    means2d_x = fx * t_view[:, 0] / t_view[:, 2] + cx
    means2d_y = fy * t_view[:, 1] / t_view[:, 2] + cy
    means2d = torch.stack([means2d_x, means2d_y], dim=-1)

    # Sort by depth
    depths = t_view[:, 2]
    idx = depths.argsort()
    means2d = means2d[idx]
    cov2d = cov2d[idx]
    colors = colors[idx]
    opacities = opacities[idx]

    # Rasterize
    pix = torch.stack(torch.meshgrid(
        torch.arange(W, device=means3D.device, dtype=torch.float32),
        torch.arange(H, device=means3D.device, dtype=torch.float32),
        indexing='xy'), dim=-1).reshape(-1, 2)

    cov_inv = cov2d.inverse()  # (N, 2, 2)
    dx = pix[:, None, :] - means2d[None, :, :]  # (HW, N, 2)

    # Mahalanobis distance
    dist2 = (dx[:, :, 0] ** 2 * cov_inv[:, 0, 0] +
             dx[:, :, 1] ** 2 * cov_inv[:, 1, 1] +
             dx[:, :, 0] * dx[:, :, 1] * (cov_inv[:, 0, 1] + cov_inv[:, 1, 0]))

    gauss = torch.exp(-0.5 * dist2) * (dist2 < 9)  # 3-sigma cutoff
    alpha = opacities[None, :, 0] * gauss  # (HW, N)

    # Alpha compositing (front-to-back)
    T = torch.cumprod(1 - alpha, dim=1)
    T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], dim=1)
    weights = T * alpha  # (HW, N)

    image = (weights[:, :, None] * colors[None, :, :]).sum(dim=1)
    return image.reshape(H, W, 3), means2d, cov2d


def draw_3d_ellipsoid(ax, mean, cov, color, alpha_surf=0.3, n_points=30):
    """Draw a 3D ellipsoid on a matplotlib 3D axis."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    radii = np.sqrt(np.abs(eigenvalues)) * 2  # 2-sigma

    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            point = eigenvectors @ np.array([x[i, j], y[i, j], z[i, j]]) + mean
            x[i, j], y[i, j], z[i, j] = point

    ax.plot_surface(x, y, z, color=color, alpha=alpha_surf, shade=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = "gs_vs_scaling_gaussians/papier_TRO_scaling_gaussians/tro_figures"
    os.makedirs(out_dir, exist_ok=True)

    # ── Define 3 ellipsoids ──
    means3D = torch.tensor([
        [-0.4, 0.0, 3.0],   # left (red)
        [0.0, 0.2, 3.5],    # center (green)
        [0.4, -0.1, 3.2],   # right (blue)
    ], device=device, dtype=torch.float32)

    base_scales = torch.tensor([
        [0.15, 0.08, 0.06],
        [0.10, 0.12, 0.05],
        [0.07, 0.10, 0.14],
    ], device=device, dtype=torch.float32)

    # Different orientations
    quats = torch.tensor([
        [1.0, 0.1, 0.2, 0.0],
        [1.0, 0.0, 0.0, 0.3],
        [1.0, -0.2, 0.1, 0.1],
    ], device=device, dtype=torch.float32)

    colors = torch.tensor([
        [0.85, 0.2, 0.2],   # red
        [0.2, 0.75, 0.2],   # green
        [0.2, 0.3, 0.85],   # blue
    ], device=device, dtype=torch.float32)

    opacities = torch.ones(3, 1, device=device)

    # Camera
    viewmat = torch.eye(4, device=device)
    viewmat = viewmat.T  # column-major
    fx, fy = 300.0, 300.0
    cx, cy = 128.0, 128.0
    W, H = 256, 256

    # ── Scale factors ──
    alpha_values = [1.0, 1.4, 1.8, 2.2, 2.6]

    # ── Figure: top row = 3D ellipsoids, bottom row = splatted images ──
    n = len(alpha_values)
    fig = plt.figure(figsize=(3.5 * n, 7))

    for i, alpha in enumerate(alpha_values):
        scales = base_scales * alpha

        # Top row: 3D view of ellipsoids
        ax3d = fig.add_subplot(2, n, i + 1, projection='3d')
        for k in range(3):
            cov3d_k = build_cov3d(scales[k:k+1], quats[k:k+1])[0].cpu().numpy()
            mean_k = means3D[k].cpu().numpy()
            color_k = colors[k].cpu().numpy()
            draw_3d_ellipsoid(ax3d, mean_k, cov3d_k, color_k, alpha_surf=0.4)
            ax3d.scatter(*mean_k, color=color_k, s=20, zorder=10)

        ax3d.set_xlim(-0.8, 0.8)
        ax3d.set_ylim(-0.5, 0.5)
        ax3d.set_zlim(2.5, 4.0)
        ax3d.set_xlabel('X', fontsize=8)
        ax3d.set_ylabel('Y', fontsize=8)
        ax3d.set_zlabel('Z', fontsize=8)
        ax3d.tick_params(labelsize=6)
        ax3d.view_init(elev=20, azim=-60)
        label = "Original" if alpha == 1.0 else f"{alpha}"
        ax3d.set_title(f"$\\alpha = $ {label}", fontsize=13, pad=5)

        # Bottom row: splatted image
        ax2d = fig.add_subplot(2, n, n + i + 1)
        with torch.no_grad():
            img, means2d, cov2d = render_gaussians(
                means3D, scales, quats, colors, opacities,
                viewmat, fx, fy, cx, cy, W, H)
        ax2d.imshow(img.cpu().numpy().clip(0, 1))
        ax2d.axis('off')

    fig.text(0.02, 0.75, '3D Gaussians', fontsize=13, rotation=90,
             va='center', fontweight='bold')
    fig.text(0.02, 0.28, 'Splatted image', fontsize=13, rotation=90,
             va='center', fontweight='bold')

    fig.suptitle("Effect of scale factor $\\alpha$ on 3D Gaussian primitives and their splatting",
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=[0.04, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "alpha_illustration.pdf"),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, "alpha_illustration.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Done: {out_dir}/alpha_illustration.pdf/png")


if __name__ == "__main__":
    main()
