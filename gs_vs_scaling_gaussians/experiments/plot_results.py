"""
Plot Results for Scale-Adaptive PVS
====================================

Generates publication-quality matplotlib figures from experiment JSON outputs.

Subcommands:
  evaluation   — Success rate bar chart per level and mode
  sweep        — Success rate & avg iterations vs scale factor
  domain       — Convergence domain radar/bar chart per mode and axis
  convergence  — Overlay convergence curves from .npz files

Usage:
    python -m gs_vs_scaling_gaussians.experiments.plot_results evaluation \
        --input logs/scale_evaluation/evaluation_results.json \
        --out_dir figures/

    python -m gs_vs_scaling_gaussians.experiments.plot_results sweep \
        --input logs/scale_sweep/sweep_results.json

    python -m gs_vs_scaling_gaussians.experiments.plot_results domain \
        --input logs/convergence_domain/convergence_domain.json

    python -m gs_vs_scaling_gaussians.experiments.plot_results convergence \
        --files logs/scale_vs/convergence_original.npz \
                logs/scale_vs/convergence_inflated.npz \
                logs/scale_vs/convergence_coarse_to_fine.npz
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Publication style
rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

MODE_COLORS = {
    "original":       "#2196F3",
    "inflated":       "#FF9800",
    "coarse_to_fine":  "#4CAF50",
    "smooth_decay":   "#9C27B0",
    "error_adaptive": "#F44336",
}
MODE_LABELS = {
    "original":       "Original",
    "inflated":       "Inflated",
    "coarse_to_fine":  "Coarse-to-Fine",
    "smooth_decay":   "Smooth Decay",
    "error_adaptive": "Error-Adaptive",
}


def _color(mode):
    return MODE_COLORS.get(mode, "#607D8B")


def _label(mode):
    return MODE_LABELS.get(mode, mode)


# ============================================================
# Subcommand: evaluation
# ============================================================
def plot_evaluation(args):
    with open(args.input) as f:
        data = json.load(f)

    trials = data["trials"]
    config = data["config"]
    modes = config["modes"]
    levels = config["levels"]

    # --- Fig 1: Success rate grouped bar chart ---
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(levels))
    width = 0.8 / len(modes)

    for i, mode in enumerate(modes):
        rates = []
        for level in levels:
            level_trials = [t for t in trials if t["level"] == level]
            n_success = sum(1 for t in level_trials if t["methods"][mode]["converged"])
            rates.append(100 * n_success / len(level_trials))
        offset = (i - len(modes) / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width * 0.9,
                      label=_label(mode), color=_color(mode), edgecolor="white", linewidth=0.5)
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{rate:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title(f"PVS Convergence Rate (scale factor = {config['scale_factor']})")
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in levels])
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(args.out_dir, "success_rate_by_level.pdf")
    fig.savefig(path)
    print(f"Saved {path}")

    # --- Fig 2: Average iterations (successful trials only) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, mode in enumerate(modes):
        avg_iters = []
        for level in levels:
            level_trials = [t for t in trials if t["level"] == level]
            successes = [t["methods"][mode]["iterations"]
                         for t in level_trials if t["methods"][mode]["converged"]]
            avg_iters.append(np.mean(successes) if successes else 0)
        offset = (i - len(modes) / 2 + 0.5) * width
        ax.bar(x + offset, avg_iters, width * 0.9,
               label=_label(mode), color=_color(mode), edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("Avg. Iterations to Converge")
    ax.set_title("Convergence Speed (successful trials)")
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in levels])
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(args.out_dir, "avg_iterations_by_level.pdf")
    fig.savefig(path)
    print(f"Saved {path}")

    # --- Fig 3: Final pose error boxplot (successful trials) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for level in levels:
        level_trials = [t for t in trials if t["level"] == level]
        t_data = []
        r_data = []
        labels = []
        for mode in modes:
            t_vals = [t["methods"][mode]["final_t_err"]
                      for t in level_trials if t["methods"][mode]["converged"]]
            r_vals = [t["methods"][mode]["final_r_err"]
                      for t in level_trials if t["methods"][mode]["converged"]]
            t_data.append(t_vals if t_vals else [0])
            r_data.append(r_vals if r_vals else [0])
            labels.append(_label(mode))

    # Use last level for boxplot
    bp1 = ax1.boxplot(t_data, labels=labels, patch_artist=True)
    for patch, mode in zip(bp1["boxes"], modes):
        patch.set_facecolor(_color(mode))
        patch.set_alpha(0.7)
    ax1.set_ylabel("Translation Error (m)")
    ax1.set_title(f"Final Pose Accuracy ({levels[-1].upper()})")
    ax1.tick_params(axis="x", rotation=30)

    bp2 = ax2.boxplot(r_data, labels=labels, patch_artist=True)
    for patch, mode in zip(bp2["boxes"], modes):
        patch.set_facecolor(_color(mode))
        patch.set_alpha(0.7)
    ax2.set_ylabel("Rotation Error (deg)")
    ax2.set_title(f"Final Pose Accuracy ({levels[-1].upper()})")
    ax2.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    path = os.path.join(args.out_dir, "final_pose_error_boxplot.pdf")
    fig.savefig(path)
    print(f"Saved {path}")

    plt.close("all")


# ============================================================
# Subcommand: sweep
# ============================================================
def plot_sweep(args):
    with open(args.input) as f:
        data = json.load(f)

    results = data["results"]
    config = data["config"]

    sfs = [r["scale_factor"] for r in results]
    rates = [r["success_rate"] for r in results]
    iters = [r["avg_iterations"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    color1 = "#2196F3"
    color2 = "#FF9800"

    ax1.plot(sfs, rates, "o-", color=color1, linewidth=2, markersize=6, label="Success Rate")
    ax1.set_xlabel("Scale Factor")
    ax1.set_ylabel("Success Rate (%)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    valid_iters = [it if not np.isnan(it) else 0 for it in iters]
    ax2.plot(sfs, valid_iters, "s--", color=color2, linewidth=2, markersize=6,
             label="Avg. Iterations")
    ax2.set_ylabel("Avg. Iterations", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(f"Scale Factor Sweep ({config['level'].upper()} perturbation, "
                  f"{config['n_trials']} trials)")
    ax1.grid(alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    path = os.path.join(args.out_dir, "scale_factor_sweep.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close("all")


# ============================================================
# Subcommand: domain
# ============================================================
def plot_domain(args):
    with open(args.input) as f:
        data = json.load(f)

    results = data["results"]
    all_axes = list(results[0]["axes"].keys())
    t_axes = [a for a in all_axes if a.startswith("t")]
    r_axes = [a for a in all_axes if a.startswith("r")]

    # --- Bar chart: translation axes ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(t_axes))
    n = len(results)
    width = 0.8 / n

    for i, r in enumerate(results):
        label = r["mode"] if r["mode"] != "inflated" else f"inflated(sf={r['scale_factor']:.1f})"
        color = _color(r["mode"])
        vals = [r["axes"][a] for a in t_axes]
        offset = (i - n / 2 + 0.5) * width
        ax1.bar(x + offset, vals, width * 0.9, label=label, color=color,
                edgecolor="white", linewidth=0.5)

    ax1.set_xlabel("Translation Axis")
    ax1.set_ylabel("Max. Convergent Displacement (m)")
    ax1.set_title("Convergence Domain — Translation")
    ax1.set_xticks(x)
    ax1.set_xticklabels(t_axes)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # --- Bar chart: rotation axes ---
    x = np.arange(len(r_axes))
    for i, r in enumerate(results):
        label = r["mode"] if r["mode"] != "inflated" else f"inflated(sf={r['scale_factor']:.1f})"
        color = _color(r["mode"])
        vals = [r["axes"][a] for a in r_axes]
        offset = (i - n / 2 + 0.5) * width
        ax2.bar(x + offset, vals, width * 0.9, label=label, color=color,
                edgecolor="white", linewidth=0.5)

    ax2.set_xlabel("Rotation Axis")
    ax2.set_ylabel("Max. Convergent Displacement (deg)")
    ax2.set_title("Convergence Domain — Rotation")
    ax2.set_xticks(x)
    ax2.set_xticklabels(r_axes)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(args.out_dir, "convergence_domain.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close("all")


# ============================================================
# Subcommand: convergence
# ============================================================
def plot_convergence(args):
    """Overlay convergence curves from .npz files."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax_err, ax_t, ax_r = axes

    for fpath in args.files:
        data = np.load(fpath, allow_pickle=True)
        mode = str(data.get("mode", os.path.basename(fpath)))
        errors = data["errors"]
        t_errs = data["pose_errors_t"]
        r_errs = data["pose_errors_r"]
        scales = data.get("scale_history", np.ones(len(errors)))
        label = _label(mode) if mode in MODE_LABELS else mode
        color = _color(mode)

        iters = np.arange(len(errors))
        ax_err.semilogy(iters, errors, color=color, linewidth=1.5, label=label)
        ax_t.plot(iters, t_errs, color=color, linewidth=1.5, label=label)
        ax_r.plot(iters, r_errs, color=color, linewidth=1.5, label=label)

    ax_err.set_xlabel("Iteration")
    ax_err.set_ylabel("Photometric Error (log)")
    ax_err.set_title("Error Convergence")
    ax_err.legend()
    ax_err.grid(alpha=0.3)

    ax_t.set_xlabel("Iteration")
    ax_t.set_ylabel("Translation Error (m)")
    ax_t.set_title("Position Error")
    ax_t.legend()
    ax_t.grid(alpha=0.3)

    ax_r.set_xlabel("Iteration")
    ax_r.set_ylabel("Rotation Error (deg)")
    ax_r.set_title("Orientation Error")
    ax_r.legend()
    ax_r.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(args.out_dir, "convergence_curves.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close("all")


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="Plot results for Scale-Adaptive PVS")
    p.add_argument("--out_dir", default="figures", help="Output directory for figures")
    sub = p.add_subparsers(dest="cmd", required=True)

    # evaluation
    s1 = sub.add_parser("evaluation", help="Plot batch evaluation results")
    s1.add_argument("--input", required=True)

    # sweep
    s2 = sub.add_parser("sweep", help="Plot scale factor sweep")
    s2.add_argument("--input", required=True)

    # domain
    s3 = sub.add_parser("domain", help="Plot convergence domain")
    s3.add_argument("--input", required=True)

    # convergence
    s4 = sub.add_parser("convergence", help="Overlay convergence curves")
    s4.add_argument("--files", nargs="+", required=True,
                    help=".npz files from scale_adaptive_vs.py")

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.cmd == "evaluation":
        plot_evaluation(args)
    elif args.cmd == "sweep":
        plot_sweep(args)
    elif args.cmd == "domain":
        plot_domain(args)
    elif args.cmd == "convergence":
        plot_convergence(args)


if __name__ == "__main__":
    main()
