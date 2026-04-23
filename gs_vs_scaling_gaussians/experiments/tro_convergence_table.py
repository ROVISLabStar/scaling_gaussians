"""
Multi-pose convergence table (Crombez TRO Table II style).
============================================================

Runs VS from N random initial poses to a fixed desired pose,
comparing methods and scale factors. Reports success/failure.

Convergence criterion: t_err < 0.01m AND r_err < 1.0°

Usage:
    python -m gs_vs_scaling_gaussians.experiments.tro_convergence_table \
        --ckpt <ckpt> --cfg <cfg> --goal_idx 10 --n_poses 20 \
        --out_dir gs_vs_scaling_gaussians/logs/tro_convergence_table

Author: Youssef ALJ (UM6P)
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gs_vs.simulators.RobotSimulator import SimulatorCamera
from gs_vs.simulators.CameraParameters import CameraParameters
from gs_vs.features.factory import create_feature
from gs_vs.datasets.colmap import Parser
from gsplat.rendering import rasterization


def load_cfg(path):
    data = {}
    with open(path) as f:
        for line in f:
            if ":" not in line: continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if k == "data_dir": data["data_dir"] = v
            elif k == "data_factor": data["data_factor"] = int(v)
            elif k == "normalize_world_space":
                data["normalize_world_space"] = v.lower() == "true"
    data.setdefault("data_factor", 1)
    data.setdefault("normalize_world_space", True)
    return data


@torch.no_grad()
def render(cMo, means, quats, scales, opacities, colors,
           sh_degree, K_np, W, H, device="cuda"):
    viewmat = torch.from_numpy(cMo).float().to(device)[None]
    Ks = torch.from_numpy(K_np).float().to(device)[None]
    renders, _, _ = rasterization(
        means=means, quats=quats, scales=scales,
        opacities=opacities, colors=colors,
        sh_degree=sh_degree, viewmats=viewmat, Ks=Ks,
        width=W, height=H, packed=True,
        render_mode="RGB+ED", camera_model="pinhole",
    )
    rgb = torch.clamp(renders[0, ..., :3], 0.0, 1.0)
    depth = renders[0, ..., 3]
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return gray, depth


def run_servo_quick(c2w_start, c2w_goal, means, quats, scales,
                    opacities, colors, sh_degree, K_np, W, H,
                    cam_params, mu, gain, max_iter, device="cuda"):
    """Run VS and return convergence info."""
    cMo_goal = np.linalg.inv(c2w_goal)
    cMo_start = np.linalg.inv(c2w_start)

    gray_des, depth_des = render(cMo_goal, means, quats, scales, opacities,
                                 colors, sh_degree, K_np, W, H, device)
    s_star = create_feature("pinhole", device=device, border=10)
    s_star.init(H, W)
    s_star.setCameraParameters(cam_params)
    s_star.buildFrom(gray_des, depth_des)

    wMo = np.eye(4)
    robot = SimulatorCamera()
    robot.setPosition(wMo @ np.linalg.inv(cMo_start))
    robot.setRobotState(1)
    cMo = cMo_start.copy()

    for it in range(max_iter):
        gray_cur, depth_cur = render(cMo, means, quats, scales, opacities,
                                      colors, sh_degree, K_np, W, H, device)
        s = create_feature("pinhole", device=device, border=10)
        s.init(H, W)
        s.setCameraParameters(cam_params)
        s.buildFrom(gray_cur, depth_cur)

        error = s.error(s_star)
        err_norm = torch.sum(error ** 2).item()

        c2w_cur = np.linalg.inv(cMo)
        t_err = np.linalg.norm(c2w_cur[:3, 3] - c2w_goal[:3, 3])
        R_rel = c2w_cur[:3, :3].T @ c2w_goal[:3, :3]
        r_err = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))

        if t_err < 0.01 and r_err < 1.0:
            return True, it, t_err, r_err, err_norm

        # Check divergence
        if err_norm > 1e8 or np.isnan(err_norm):
            return False, it, t_err, r_err, err_norm

        Ls = s.interaction()
        Hs = Ls.T @ Ls
        diagHs = torch.diag(torch.diag(Hs))
        Hess = torch.linalg.inv(mu * diagHs + Hs + 1e-6 * torch.eye(6, device=device))
        v = -gain * (Hess @ Ls.T @ error)
        v_np = v.detach().cpu().numpy()

        vt = np.linalg.norm(v_np[:3])
        vr = np.linalg.norm(v_np[3:])
        if vt > 0.5: v_np[:3] *= 0.5 / vt
        if vr > 0.3: v_np[3:] *= 0.3 / vr

        robot.setVelocity("camera", v_np)
        wMc = robot.getPosition()
        cMo = np.linalg.inv(wMc) @ wMo

    return False, max_iter, t_err, r_err, err_norm


def main():
    p = argparse.ArgumentParser(description="TRO Convergence Table")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--data_factor", type=int, default=8)
    p.add_argument("--goal_idx", type=int, default=10)
    p.add_argument("--n_poses", type=int, default=20)
    p.add_argument("--max_iter", type=int, default=500)
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--gain", type=float, default=10.0)
    p.add_argument("--out_dir", type=str,
                   default="gs_vs_scaling_gaussians/logs/tro_convergence_table")
    args = p.parse_args()
    device = "cuda"

    # Load scene
    cfg = load_cfg(args.cfg)
    parser = Parser(data_dir=cfg["data_dir"], factor=cfg["data_factor"],
                    normalize=cfg["normalize_world_space"], test_every=8)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    splats = ckpt["splats"]
    means = splats["means"].to(device)
    quats = splats["quats"].to(device)
    scales_orig = torch.exp(splats["scales"]).to(device)
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

    c2w_goal = camtoworlds[args.goal_idx]
    os.makedirs(args.out_dir, exist_ok=True)

    # Select N starting poses within reasonable distance of goal
    # Filter: translation 0.05-0.6m, rotation < 50° (within convergence range)
    np.random.seed(42)
    goal_pos = c2w_goal[:3, 3]
    goal_R = c2w_goal[:3, :3]

    candidates = []
    for i in range(len(camtoworlds)):
        if i == args.goal_idx:
            continue
        c2w_i = camtoworlds[i]
        t_dist = np.linalg.norm(c2w_i[:3, 3] - goal_pos)
        R_rel = c2w_i[:3, :3].T @ goal_R
        r_dist = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
        if 0.05 < t_dist < 0.6 and r_dist < 50:
            candidates.append((i, t_dist, r_dist))

    print(f"[Pose selection] {len(candidates)} candidates within "
          f"t∈[0.05, 0.6]m, r<50° of goal")

    if len(candidates) < args.n_poses:
        print(f"  WARNING: only {len(candidates)} candidates, using all")
        start_indices = sorted([c[0] for c in candidates])
    else:
        selected = sorted(candidates, key=lambda x: x[1])  # sort by distance
        # Sample evenly across distances
        step = len(selected) // args.n_poses
        picked = [selected[i * step] for i in range(args.n_poses)]
        start_indices = sorted([c[0] for c in picked])

    args.n_poses = len(start_indices)

    # Methods to compare
    methods = {
        "Original": 1.0,
        "α=1.6": 1.6,
        "α=1.8": 1.8,
        "α=2.0": 2.0,
        "α=2.2": 2.2,
        "α=2.4": 2.4,
    }

    print(f"[Convergence Table] Goal: view {args.goal_idx}")
    print(f"[Convergence Table] {args.n_poses} start poses, "
          f"{len(methods)} methods, max_iter={args.max_iter}")
    print()

    # Results table: rows = poses, cols = methods
    results = {}  # (pose_idx, method_name) → (converged, n_iter, t_err, r_err)

    for pose_num, start_idx in enumerate(start_indices):
        c2w_start = camtoworlds[start_idx]
        t_init = np.linalg.norm(c2w_start[:3, 3] - c2w_goal[:3, 3])
        R_rel = c2w_start[:3, :3].T @ c2w_goal[:3, :3]
        r_init = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))

        print(f"Pose {pose_num+1:>2}/{args.n_poses} "
              f"(idx={start_idx:>3}, t={t_init:.3f}m, r={r_init:.1f}°) ",
              end="", flush=True)

        for method_name, sf in methods.items():
            scales = scales_orig * sf
            conv, n_it, t_err, r_err, cost = run_servo_quick(
                c2w_start, c2w_goal, means, quats, scales,
                opacities, colors, sh_degree, K_np, W, H,
                cam_params, args.mu, args.gain, args.max_iter, device)

            results[(start_idx, method_name)] = {
                "converged": conv,
                "iterations": n_it,
                "t_err": t_err,
                "r_err": r_err,
                "cost": cost,
            }
            mark = "✓" if conv else "✗"
            print(f" {mark}", end="", flush=True)

        print()

    # ─── Print Table ───
    print("\n" + "=" * 80)
    header = f"{'Pose':>6} {'Idx':>4} {'t₀(m)':>7} {'r₀(°)':>7}"
    for m in methods:
        header += f" {m:>8}"
    print(header)
    print("-" * 80)

    method_success = {m: 0 for m in methods}

    for pose_num, start_idx in enumerate(start_indices):
        c2w_start = camtoworlds[start_idx]
        t_init = np.linalg.norm(c2w_start[:3, 3] - c2w_goal[:3, 3])
        R_rel = c2w_start[:3, :3].T @ c2w_goal[:3, :3]
        r_init = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))

        row = f"{pose_num+1:>6} {start_idx:>4} {t_init:>7.3f} {r_init:>7.1f}"
        for m in methods:
            r = results[(start_idx, m)]
            if r["converged"]:
                row += f" {'✓':>8}"
                method_success[m] += 1
            else:
                row += f" {'✗':>8}"
        print(row)

    print("-" * 80)
    total_row = f"{'':>6} {'':>4} {'':>7} {'Total':>7}"
    for m in methods:
        total_row += f" {method_success[m]:>5}/{args.n_poses:<2}"
    print(total_row)
    print("=" * 80)

    # ─── Save LaTeX table ───
    latex_lines = []
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Multi-pose convergence comparison. "
                       "Successful ($\\checkmark$) and failed ($\\times$) "
                       f"convergences for {args.n_poses} random initial poses "
                       f"(goal: view {args.goal_idx}).}}")
    latex_lines.append("\\label{tab:convergence}")

    cols = "r" * 2 + "|" + "c" * len(methods)
    latex_lines.append(f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}")
    latex_lines.append("\\toprule")

    # Header
    header_tex = " & ".join(["Pose", "$\\Delta$"] +
                            [f"\\textbf{{{m}}}" for m in methods])
    latex_lines.append(header_tex + " \\\\")
    latex_lines.append("\\midrule")

    for pose_num, start_idx in enumerate(start_indices):
        c2w_start = camtoworlds[start_idx]
        t_init = np.linalg.norm(c2w_start[:3, 3] - c2w_goal[:3, 3])
        R_rel = c2w_start[:3, :3].T @ c2w_goal[:3, :3]
        r_init = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))

        disp = f"{t_init:.2f}m, {r_init:.0f}°"
        cells = [f"{pose_num+1}", disp]
        for m in methods:
            r = results[(start_idx, m)]
            cells.append("$\\checkmark$" if r["converged"] else "$\\times$")
        latex_lines.append(" & ".join(cells) + " \\\\")

    latex_lines.append("\\midrule")
    total_cells = ["", "\\textbf{Total}"]
    for m in methods:
        total_cells.append(f"\\textbf{{{method_success[m]}/{args.n_poses}}}")
    latex_lines.append(" & ".join(total_cells) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    latex_path = os.path.join(args.out_dir, "convergence_table.tex")
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"\nLaTeX table: {latex_path}")

    # ─── Bar chart ───
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    x = np.arange(len(methods))
    counts = [method_success[m] for m in methods]
    colors_bar = ["#d62728", "#ff7f0e", "#9467bd", "#1f77b4", "#8c564b"]
    bars = ax.bar(x, counts, color=colors_bar, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(list(methods.keys()), fontsize=10)
    ax.set_ylabel(f"Successful convergences (/{args.n_poses})", fontsize=11)
    ax.set_ylim(0, args.n_poses + 1)
    ax.axhline(y=args.n_poses, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, axis='y', alpha=0.3)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(count), ha='center', fontsize=10, fontweight='bold')

    ax.set_title(f"Convergence Rate ({args.n_poses} random poses → view {args.goal_idx})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "convergence_bar.pdf"), dpi=300)
    fig.savefig(os.path.join(args.out_dir, "convergence_bar.png"), dpi=200)
    plt.close(fig)

    # ─── Save raw results ───
    import json
    raw = {
        "goal_idx": args.goal_idx,
        "start_indices": [int(x) for x in start_indices],
        "methods": list(methods.keys()),
        "scale_factors": list(methods.values()),
        "max_iter": args.max_iter,
        "results": {f"{si}_{m}": r for (si, m), r in results.items()},
        "success_counts": method_success,
    }
    with open(os.path.join(args.out_dir, "convergence_results.json"), "w") as f:
        json.dump(raw, f, indent=2, default=str)

    print(f"Figures and data saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
