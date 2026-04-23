"""
Spherical Decoupling Visualization for Photometric Visual Servoing
===================================================================

Visualizes on a unit sphere (via viser) why spherical projection decouples
rotation from translation in PVS:

  - ROTATION PURE  → rigid displacement on the sphere (uniform flow)
  - TRANSLATION PURE → depth-dependent flow with focus of expansion
  - COMBINED → superposition

Usage:
    python sphere_decoupling_viz.py --port 8080
    python sphere_decoupling_viz.py --ckpt /path/to/ckpt.pt --port 8080

Author: Youssef (UM6P / Ai Movement Lab)
"""

import argparse
import numpy as np
import torch
import viser
import viser.transforms as vtf
import time
from scipy.spatial.transform import Rotation as Rot


# ─────────────────────────────────────────────
# 1. GEOMETRIC UTILITIES
# ─────────────────────────────────────────────

def apply_rotation(points_on_sphere, R_mat):
    return (R_mat @ points_on_sphere.T).T


def compute_spherical_flow(s_current, s_desired):
    dot = np.sum(s_current * s_desired, axis=-1, keepdims=True)
    tangent = s_desired - dot * s_current
    return tangent


# ─────────────────────────────────────────────
# 2. SCENE GENERATION
# ─────────────────────────────────────────────

def generate_synthetic_scene(n_points=2000, seed=42):
    rng = np.random.default_rng(seed)
    pts = np.zeros((n_points, 3))
    pts[:, 0] = rng.uniform(-2.0, 2.0, n_points)
    pts[:, 1] = rng.uniform(-1.5, 1.5, n_points)
    pts[:, 2] = rng.uniform(1.5, 6.0, n_points)

    depth_norm = (pts[:, 2] - 1.5) / 4.5
    colors = np.zeros((n_points, 3))
    colors[:, 0] = 0.2 + 0.8 * (1 - depth_norm)
    colors[:, 1] = 0.3
    colors[:, 2] = 0.2 + 0.8 * depth_norm
    return pts, colors


def generate_scene_from_gsplat(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "splats" in ckpt:
            splats = ckpt["splats"]
        else:
            splats = ckpt

        if "means3d" in splats:
            means = splats["means3d"].numpy()
        elif "means" in splats:
            means = splats["means"].numpy()
        else:
            print("[WARN] No means found. Using synthetic scene.")
            return generate_synthetic_scene()

        if len(means) > 5000:
            idx = np.random.choice(len(means), 5000, replace=False)
            means = means[idx]
        else:
            idx = np.arange(len(means))

        if "sh0" in splats:
            sh0 = splats["sh0"].numpy()
            if len(sh0) > 5000:
                sh0 = sh0[idx]
            colors = np.clip(0.5 + 0.28 * sh0.squeeze(-2), 0, 1)
        else:
            depth = np.linalg.norm(means, axis=-1)
            dn = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            colors = np.stack([1 - dn, 0.3 * np.ones_like(dn), dn], axis=-1)

        print(f"[INFO] Loaded {len(means)} Gaussians from checkpoint")
        return means, colors

    except Exception as e:
        print(f"[WARN] Could not load checkpoint ({e}). Using synthetic scene.")
        return generate_synthetic_scene()


# ─────────────────────────────────────────────
# 3. VISER VISUALIZATION
# ─────────────────────────────────────────────

def build_sphere_wireframe(server, radius=1.0, n_meridians=12, n_parallels=8,
                           color=(80, 80, 100), line_width=1.0, name="sphere"):
    lines_start = []
    lines_end = []

    for i in range(n_meridians):
        phi = 2.0 * np.pi * i / n_meridians
        for j in range(60):
            theta1 = np.pi * j / 60
            theta2 = np.pi * (j + 1) / 60
            lines_start.append([
                radius * np.sin(theta1) * np.cos(phi),
                radius * np.sin(theta1) * np.sin(phi),
                radius * np.cos(theta1)
            ])
            lines_end.append([
                radius * np.sin(theta2) * np.cos(phi),
                radius * np.sin(theta2) * np.sin(phi),
                radius * np.cos(theta2)
            ])

    for j in range(1, n_parallels):
        theta = np.pi * j / n_parallels
        for i in range(60):
            phi1 = 2.0 * np.pi * i / 60
            phi2 = 2.0 * np.pi * (i + 1) / 60
            lines_start.append([
                radius * np.sin(theta) * np.cos(phi1),
                radius * np.sin(theta) * np.sin(phi1),
                radius * np.cos(theta)
            ])
            lines_end.append([
                radius * np.sin(theta) * np.cos(phi2),
                radius * np.sin(theta) * np.sin(phi2),
                radius * np.cos(theta)
            ])

    lines_start = np.array(lines_start, dtype=np.float32)
    lines_end = np.array(lines_end, dtype=np.float32)

    server.scene.add_line_segments(
        f"/{name}/wireframe",
        points=np.stack([lines_start, lines_end], axis=1),  # (N, 2, 3)
        colors=np.array(color, dtype=np.uint8),              # (3,)
        line_width=line_width,
    )


def add_flow_arrows(server, origins, flows, color, name, scale=2.0, max_arrows=500):
    if len(origins) > max_arrows:
        idx = np.linspace(0, len(origins) - 1, max_arrows, dtype=int)
        origins = origins[idx]
        flows = flows[idx]

    flow_norms = np.linalg.norm(flows, axis=-1)
    mask = flow_norms > 1e-6
    origins = origins[mask]
    flows = flows[mask]

    if len(origins) == 0:
        return

    endpoints = origins + flows * scale

    server.scene.add_line_segments(
        f"/flow/{name}/lines",
        points=np.stack([origins, endpoints], axis=1).astype(np.float32),  # (N, 2, 3)
        colors=np.array(color, dtype=np.uint8),                            # (3,)
        line_width=2.0,
    )

    server.scene.add_point_cloud(
        f"/flow/{name}/tips",
        points=endpoints.astype(np.float32),
        colors=np.tile(np.array(color, dtype=np.uint8), (len(endpoints), 1)),
        point_size=0.02,
        point_shape="circle",
    )


# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Spherical decoupling visualization")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--n_points", type=int, default=3000)
    parser.add_argument("--rot_deg", type=float, default=15.0)
    parser.add_argument("--trans_m", type=float, default=0.5)
    args = parser.parse_args()

    if args.ckpt:
        pts_3d, colors = generate_scene_from_gsplat(args.ckpt)
    else:
        pts_3d, colors = generate_synthetic_scene(n_points=args.n_points)

    mask_visible = pts_3d[:, 2] > 0.1
    pts_visible = pts_3d[mask_visible]
    colors_visible = colors[mask_visible]

    depths = np.linalg.norm(pts_visible, axis=-1, keepdims=True)
    s_current = pts_visible / depths

    # Precompute flows for analysis printout
    R_pure = Rot.from_rotvec([0, np.deg2rad(args.rot_deg), 0]).as_matrix()
    s_after_rot = apply_rotation(s_current, R_pure)
    flow_rot = compute_spherical_flow(s_current, s_after_rot)

    t_pure = np.array([args.trans_m, 0, 0])
    pts_after_trans = pts_visible + t_pure[None, :]
    s_after_trans = pts_after_trans / np.linalg.norm(pts_after_trans, axis=-1, keepdims=True)
    flow_trans = compute_spherical_flow(s_current, s_after_trans)

    # Launch viser
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"\n{'='*60}")
    print(f"  SPHERICAL DECOUPLING VISUALIZATION")
    print(f"  Open: http://localhost:{args.port}")
    print(f"{'='*60}\n")

    # GUI
    with server.gui.add_folder("Display"):
        show_sphere = server.gui.add_checkbox("Show sphere wireframe", initial_value=True)
        show_points = server.gui.add_checkbox("Show scene points on sphere", initial_value=True)
        sphere_radius = server.gui.add_slider("Sphere radius", min=0.5, max=3.0, step=0.1, initial_value=1.0)

    with server.gui.add_folder("Flow Fields"):
        show_rot = server.gui.add_checkbox("Rotation flow (RED)", initial_value=True)
        show_trans = server.gui.add_checkbox("Translation flow (BLUE)", initial_value=True)
        show_combined = server.gui.add_checkbox("Combined flow (GREEN)", initial_value=False)
        arrow_scale = server.gui.add_slider("Arrow scale", min=0.5, max=10.0, step=0.5, initial_value=3.0)
        max_arrows_slider = server.gui.add_slider("Max arrows", min=50, max=1000, step=50, initial_value=400)

    with server.gui.add_folder("Motion Parameters"):
        rot_slider = server.gui.add_slider("Rotation (deg)", min=1.0, max=45.0, step=1.0,
                                            initial_value=args.rot_deg)
        trans_slider = server.gui.add_slider("Translation (m)", min=0.05, max=2.0, step=0.05,
                                              initial_value=args.trans_m)

    with server.gui.add_folder("Analysis"):
        server.gui.add_markdown(
            "**Key insight:**\n"
            "- **Red arrows (rotation)** = uniform length, parallel flow = **depth-independent**\n"
            "- **Blue arrows (translation)** = varying length = **depth-dependent**\n"
            "- On the sphere, rotation is a rigid motion => trivially decoupled\n"
            "- The interaction matrix L_s for spherical features separates v and omega"
        )

    def render():
        R_val = sphere_radius.value
        angle = np.deg2rad(rot_slider.value)
        R_rot = Rot.from_rotvec([0, angle, 0]).as_matrix()
        t_tr = np.array([trans_slider.value, 0, 0])

        s_cur = s_current * R_val

        s_rot = apply_rotation(s_current, R_rot) * R_val
        f_rot = s_rot - s_cur

        pts_t = pts_visible + t_tr[None, :]
        d_t = np.linalg.norm(pts_t, axis=-1, keepdims=True)
        s_tr = (pts_t / d_t) * R_val
        f_trans = s_tr - s_cur

        pts_c = (R_rot @ pts_visible.T).T + t_tr[None, :]
        d_c = np.linalg.norm(pts_c, axis=-1, keepdims=True)
        s_comb = (pts_c / d_c) * R_val
        f_comb = s_comb - s_cur

        if show_sphere.value:
            build_sphere_wireframe(server, radius=R_val, name="sphere")

        if show_points.value:
            server.scene.add_point_cloud(
                "/points/on_sphere",
                points=s_cur.astype(np.float32),
                colors=(colors_visible * 255).astype(np.uint8),
                point_size=0.01,
                point_shape="circle",
            )

        sc = arrow_scale.value
        ma = int(max_arrows_slider.value)

        if show_rot.value:
            add_flow_arrows(server, s_cur, f_rot, color=(230, 60, 60),
                          name="rotation", scale=sc, max_arrows=ma)
        if show_trans.value:
            add_flow_arrows(server, s_cur, f_trans, color=(60, 100, 230),
                          name="translation", scale=sc, max_arrows=ma)
        if show_combined.value:
            add_flow_arrows(server, s_cur, f_comb, color=(60, 200, 80),
                          name="combined", scale=sc, max_arrows=ma)

        server.scene.add_frame("/origin", axes_length=0.3, axes_radius=0.01)

        if show_trans.value:
            foe_dir = t_tr / (np.linalg.norm(t_tr) + 1e-8)
            foe_point = foe_dir * R_val
            server.scene.add_point_cloud(
                "/flow/translation/foe",
                points=foe_point.reshape(1, 3).astype(np.float32),
                colors=np.array([[255, 255, 0]], dtype=np.uint8),
                point_size=0.06,
                point_shape="diamond",
            )
            server.scene.add_label(
                "/flow/translation/foe_label",
                text="FoE",
                wxyz=vtf.SO3.identity().wxyz,
                position=foe_point * 1.15,
            )

    render()

    @show_sphere.on_update
    def _(_): render()
    @show_points.on_update
    def _(_): render()
    @show_rot.on_update
    def _(_): render()
    @show_trans.on_update
    def _(_): render()
    @show_combined.on_update
    def _(_): render()
    @arrow_scale.on_update
    def _(_): render()
    @rot_slider.on_update
    def _(_): render()
    @trans_slider.on_update
    def _(_): render()
    @sphere_radius.on_update
    def _(_): render()
    @max_arrows_slider.on_update
    def _(_): render()

    # Analysis printout
    rot_norms = np.linalg.norm(flow_rot, axis=-1)
    trans_norms = np.linalg.norm(flow_trans, axis=-1)

    print(f"--- Flow Field Analysis ---")
    print(f"Rotation flow norm:    mean={rot_norms.mean():.4f}, "
          f"std={rot_norms.std():.4f}, CV={rot_norms.std()/rot_norms.mean():.4f}")
    print(f"Translation flow norm: mean={trans_norms.mean():.4f}, "
          f"std={trans_norms.std():.4f}, CV={trans_norms.std()/trans_norms.mean():.4f}")
    print(f"\n-> Rotation CV ~ 0 confirms depth-independence (rigid motion on S2)")
    print(f"-> Translation CV >> 0 confirms depth-dependence\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
