#!/usr/bin/env python3
"""
Main entry point for systematic photometric visual servoing evaluation.

Usage:

  # Run the projection-consistency experiment (all 5 models, 30 trials)
  python examples/run_evaluation.py \\
      --ckpt /path/to/ckpt_6999_rank0.pt \\
      --cfg  /path/to/cfg.yml \\
      --experiment projection_consistency \\
      --n_trials 30 \\
      --save_dir results/projection_consistency

  # Quick test with 3 trials, only pinhole vs equidistant
  python examples/run_evaluation.py \\
      --ckpt /path/to/ckpt_6999_rank0.pt \\
      --cfg  /path/to/cfg.yml \\
      --experiment projection_consistency \\
      --models pinhole equidistant \\
      --n_trials 3 \\
      --verbose

  # Custom experiment: compare unified variants on large perturbations
  python examples/run_evaluation.py \\
      --ckpt /path/to/ckpt_6999_rank0.pt \\
      --cfg  /path/to/cfg.yml \\
      --experiment unified_variants \\
      --perturbation large \\
      --n_trials 30 \\
      --save_dir results/unified_large

  # Sweep across perturbation levels
  python examples/run_evaluation.py \\
      --ckpt /path/to/ckpt_6999_rank0.pt \\
      --cfg  /path/to/cfg.yml \\
      --experiment sweep_perturbations \\
      --models pinhole equidistant \\
      --save_dir results/sweep
"""

import argparse
import sys
import os
import torch
import numpy as np

sys.path.append("./external/_gsplat")
sys.path.append("./external/_gsplat/examples")

from evaluation.configs import (
    ExperimentConfig, ServoParams,
    FISHEYE_1024, FISHEYE_640, PINHOLE_DEFAULT,
    PERTURB_SMALL, PERTURB_MEDIUM, PERTURB_LARGE,
    PERTURB_TRANSLATION_ONLY, PERTURB_ROTATION_ONLY,
    INTERACTION_MODELS,
    projection_consistency_experiment,
    unified_variant_experiment,
    convergence_basin_experiment,
    pinhole_baseline_experiment,
)
from evaluation.batch_runner import run_experiment
from evaluation.metrics import summarize_trials, print_summary_table
from evaluation.plotting import generate_all_plots, generate_latex_table


# ======================================================================
# GS model loading
# ======================================================================

def load_gs_model(ckpt_path: str, cfg_path: str, device: str = "cuda"):
    """
    Load Gaussian Splatting model and dataset parser.

    Returns:
        splat_data: dict with means, quats, scales, opacities, colors, sh_degree
        parser: dataset Parser object
    """
    from datasets.colmap import Parser

    # Load config
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

    parser = Parser(
        data_dir=data["data_dir"],
        factor=data["data_factor"],
        normalize=data["normalize_world_space"],
        test_every=8,
    )

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    splats = ckpt["splats"]

    splat_data = {
        "means": splats["means"].to(device),
        "quats": splats["quats"].to(device),
        "scales": torch.exp(splats["scales"]).to(device),
        "opacities": torch.sigmoid(splats["opacities"]).to(device),
        "colors": torch.cat([splats["sh0"], splats["shN"]], dim=1).to(device),
        "sh_degree": int(np.sqrt(
            torch.cat([splats["sh0"], splats["shN"]], dim=1).shape[1]
        ) - 1),
    }

    print(f"Loaded {splat_data['means'].shape[0]} Gaussians "
          f"(SH degree {splat_data['sh_degree']})")

    return splat_data, parser


# ======================================================================
# Experiment builders
# ======================================================================

PERTURBATION_MAP = {
    "small": PERTURB_SMALL,
    "medium": PERTURB_MEDIUM,
    "large": PERTURB_LARGE,
    "trans_only": PERTURB_TRANSLATION_ONLY,
    "rot_only": PERTURB_ROTATION_ONLY,
}

CAMERA_MAP = {
    "fisheye_1024": FISHEYE_1024,
    "fisheye_640": FISHEYE_640,
    "pinhole": PINHOLE_DEFAULT,
}

EXPERIMENT_MAP = {
    "projection_consistency": projection_consistency_experiment,
    "unified_variants": unified_variant_experiment,
    "convergence_basin": convergence_basin_experiment,
    "pinhole_baseline": pinhole_baseline_experiment,
    # "convergence_basin_grid" is handled separately below (not a standard batch run)
}


def build_experiment(args) -> list:
    """
    Build experiment config(s) from CLI arguments.

    Returns a list of ExperimentConfig objects.
    """
    if args.experiment == "sweep_perturbations":
        # Special: run the same models across small/medium/large
        configs = []
        for level_name, perturb in [
            ("small", PERTURB_SMALL),
            ("medium", PERTURB_MEDIUM),
            ("large", PERTURB_LARGE),
        ]:
            camera = CAMERA_MAP.get(args.camera, FISHEYE_1024)
            models = args.models or ["pinhole", "equidistant"]
            configs.append(ExperimentConfig(
                name=f"sweep_{level_name}",
                camera=camera,
                perturbation=perturb,
                servo=ServoParams(
                    max_iter=args.max_iter,
                    convergence_thresh=args.convergence_thresh,
                ),
                interaction_models=models,
                n_trials=args.n_trials,
                seed=args.seed,
            ))
        return configs

    # Standard experiment
    if args.experiment in EXPERIMENT_MAP:
        config = EXPERIMENT_MAP[args.experiment]()
    else:
        # Custom experiment
        camera = CAMERA_MAP.get(args.camera, FISHEYE_1024)
        perturb = PERTURBATION_MAP.get(args.perturbation, PERTURB_MEDIUM)
        models = args.models or list(INTERACTION_MODELS.keys())
        config = ExperimentConfig(
            name=args.experiment or "custom",
            camera=camera,
            perturbation=perturb,
            servo=ServoParams(),
            interaction_models=models,
            n_trials=args.n_trials,
            seed=args.seed,
        )

    # Override from CLI
    if args.models:
        config.interaction_models = args.models
    if args.n_trials:
        config.n_trials = args.n_trials
    if args.perturbation:
        config.perturbation = PERTURBATION_MAP.get(
            args.perturbation, config.perturbation
        )
    if args.camera:
        config.camera = CAMERA_MAP.get(args.camera, config.camera)
    config.servo.max_iter = args.max_iter
    config.servo.convergence_thresh = args.convergence_thresh
    config.seed = args.seed

    return [config]


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Systematic evaluation of photometric visual servoing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument("--ckpt", required=True, help="GS checkpoint path")
    parser.add_argument("--cfg", required=True, help="GS config (cfg.yml)")

    # Experiment selection
    parser.add_argument(
        "--experiment", default="projection_consistency",
        choices=list(EXPERIMENT_MAP.keys()) + [
            "sweep_perturbations", "convergence_basin_grid", "cost_surface", "custom"
        ],
        help="Predefined experiment name",
    )

    # Overrides
    parser.add_argument(
        "--models", nargs="+",
        choices=list(INTERACTION_MODELS.keys()),
        help="Override interaction models to evaluate",
    )
    parser.add_argument(
        "--perturbation",
        choices=list(PERTURBATION_MAP.keys()),
        help="Override perturbation level",
    )
    parser.add_argument(
        "--camera",
        choices=list(CAMERA_MAP.keys()),
        help="Override camera preset",
    )
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=4000)
    parser.add_argument("--convergence_thresh", type=float, default=100.0)
    parser.add_argument("--desired_pose_idx", type=int, default=100,
                        help="Dataset pose index for desired image")

    # Output
    parser.add_argument("--save_dir", default=None,
                        help="Directory to save results and plots")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--debug_images", action="store_true",
                        help="Save [current|desired|diff] debug images (first trial per model)")
    parser.add_argument("--debug_save_interval", type=int, default=50,
                        help="Save debug image every N iterations")

    # Convergence basin grid options
    parser.add_argument("--basin_dof_x", default="tx",
                        choices=["tx", "ty", "tz", "rx", "ry", "rz"],
                        help="X-axis DOF for basin grid")
    parser.add_argument("--basin_dof_y", default="tz",
                        choices=["tx", "ty", "tz", "rx", "ry", "rz"],
                        help="Y-axis DOF for basin grid")
    parser.add_argument("--basin_range", type=float, default=0.3,
                        help="Symmetric range for basin grid (±value)")
    parser.add_argument("--basin_resolution", type=int, default=15,
                        help="Grid points per axis for basin analysis")

    args = parser.parse_args()

    # ---- Load GS model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splat_data, dataset_parser = load_gs_model(args.ckpt, args.cfg, device)

    # ---- Special case: convergence basin grid ----
    if args.experiment == "convergence_basin_grid":
        from evaluation.convergence_basin import (
            run_convergence_basin,
            plot_convergence_basin_heatmaps,
            plot_convergence_basin_overlay,
        )
        from evaluation.configs import (
            FISHEYE_1024, ServoParams, get_valid_models,
            validate_experiment_models,
        )
        from evaluation.batch_runner import setup_camera_and_mask

        camera = CAMERA_MAP.get(args.camera, FISHEYE_1024)
        models = args.models or get_valid_models(
            camera.render_model, include_ablation=True
        )
        validate_experiment_models(camera.render_model, models)

        # Setup camera
        parser_K = list(dataset_parser.Ks_dict.values())[0]
        parser_W, parser_H = list(dataset_parser.imsize_dict.values())[0]

        # Build a minimal config for setup_camera_and_mask
        from evaluation.configs import ExperimentConfig, PERTURB_MEDIUM
        temp_config = ExperimentConfig(
            name="basin", camera=camera,
            perturbation=PERTURB_MEDIUM,
            servo=ServoParams(max_iter=args.max_iter,
                              convergence_thresh=args.convergence_thresh),
            interaction_models=models,
        )
        K_np, W, H, mask = setup_camera_and_mask(
            temp_config, parser_K, parser_W, parser_H, device=device,
        )

        # Desired pose
        camtoworlds = dataset_parser.camtoworlds
        pose_idx = min(args.desired_pose_idx, camtoworlds.shape[0] - 1)
        cdMo = np.linalg.inv(camtoworlds[pose_idx])

        r = args.basin_range
        vals_x, vals_y, basin_results = run_convergence_basin(
            splat_data=splat_data,
            K_np=K_np, W=W, H=H,
            render_model=camera.render_model,
            mask=mask,
            cdMo=cdMo,
            interaction_models=models,
            dof_x=args.basin_dof_x,
            dof_y=args.basin_dof_y,
            range_x=(-r, r),
            range_y=(-r, r),
            n_x=args.basin_resolution,
            n_y=args.basin_resolution,
            servo_params=temp_config.servo,
            device=device,
            verbose=args.verbose,
        )

        if args.save_dir:
            plot_dir = os.path.join(args.save_dir, "plots")
            plot_convergence_basin_heatmaps(
                vals_x, vals_y, basin_results,
                dof_x=args.basin_dof_x, dof_y=args.basin_dof_y,
                save_path=os.path.join(
                    plot_dir,
                    f"basin_{args.basin_dof_x}_{args.basin_dof_y}_heatmaps.pdf"
                ),
                title=f"Convergence Basin (render={camera.render_model})",
            )
            plot_convergence_basin_overlay(
                vals_x, vals_y, basin_results,
                dof_x=args.basin_dof_x, dof_y=args.basin_dof_y,
                save_path=os.path.join(
                    plot_dir,
                    f"basin_{args.basin_dof_x}_{args.basin_dof_y}_overlay.pdf"
                ),
                title=f"Convergence Boundaries (render={camera.render_model})",
            )

            # Save raw grid data
            np.savez(
                os.path.join(args.save_dir, "basin_data.npz"),
                vals_x=vals_x, vals_y=vals_y,
                **{f"grid_{m}": g for m, g in basin_results.items()},
            )
            print(f"\nBasin data saved to: {args.save_dir}/")

        return  # done, skip normal experiment loop

    # ---- Special case: cost surface (3D plots of ||e||^2) ----
    if args.experiment == "cost_surface":
        from evaluation.cost_surface import (
            compute_cost_surface,
            plot_cost_surface_3d,
            plot_cost_surfaces_grid,
        )
        from evaluation.configs import FISHEYE_1024, get_valid_models
        from evaluation.batch_runner import setup_camera_and_mask
        from evaluation.configs import ExperimentConfig, PERTURB_MEDIUM, ServoParams

        camera = CAMERA_MAP.get(args.camera, FISHEYE_1024)

        temp_config = ExperimentConfig(
            name="cost_surface", camera=camera,
            perturbation=PERTURB_MEDIUM,
            servo=ServoParams(),
            interaction_models=[],  # not needed for cost surface
        )
        parser_K = list(dataset_parser.Ks_dict.values())[0]
        parser_W, parser_H = list(dataset_parser.imsize_dict.values())[0]
        K_np, W, H, mask = setup_camera_and_mask(
            temp_config, parser_K, parser_W, parser_H, device=device,
        )

        camtoworlds = dataset_parser.camtoworlds
        pose_idx = min(args.desired_pose_idx, camtoworlds.shape[0] - 1)
        cdMo = np.linalg.inv(camtoworlds[pose_idx])

        r = args.basin_range
        n = args.basin_resolution

        print(f"\nComputing cost surface: "
              f"{args.basin_dof_x} vs {args.basin_dof_y}, "
              f"range=±{r}, resolution={n}x{n}")

        vals_x, vals_y, cost_grid = compute_cost_surface(
            splat_data=splat_data,
            K_np=K_np, W=W, H=H,
            render_model=camera.render_model,
            mask=mask,
            cdMo=cdMo,
            dof_x=args.basin_dof_x,
            dof_y=args.basin_dof_y,
            range_x=(-r, r),
            range_y=(-r, r),
            n_x=n, n_y=n,
            device=device,
            verbose=args.verbose,
        )

        if args.save_dir:
            plot_dir = os.path.join(args.save_dir, "plots")
            plot_cost_surface_3d(
                vals_x, vals_y, cost_grid,
                dof_x=args.basin_dof_x,
                dof_y=args.basin_dof_y,
                title=f"Cost Surface ({camera.render_model})",
                save_path=os.path.join(
                    plot_dir,
                    f"cost_surface_{args.basin_dof_x}_{args.basin_dof_y}.pdf"
                ),
            )
            np.savez(
                os.path.join(args.save_dir, "cost_surface.npz"),
                vals_x=vals_x, vals_y=vals_y, cost=cost_grid,
            )
            print(f"Cost surface saved to: {args.save_dir}/")

        return  # done

    # ---- Build experiment(s) ----
    configs = build_experiment(args)

    # ---- Run ----
    all_summaries = []

    for config in configs:
        results = run_experiment(
            config=config,
            splat_data=splat_data,
            parser=dataset_parser,
            desired_pose_index=args.desired_pose_idx,
            device=device,
            verbose=args.verbose,
            save_dir=args.save_dir,
            debug_images=args.debug_images,
            debug_save_interval=args.debug_save_interval,
        )

        # Summaries
        summaries = [summarize_trials(trials) for trials in results.values()]
        all_summaries.extend(summaries)

        # Generate plots
        if not args.no_plots and args.save_dir:
            plot_dir = os.path.join(args.save_dir, "plots")
            generate_all_plots(
                results, summaries,
                save_dir=plot_dir,
                experiment_name=config.name,
            )

            # Also print LaTeX
            latex = generate_latex_table(summaries)
            print(f"\n{'=' * 70}")
            print("LATEX TABLE")
            print('=' * 70)
            print(latex)

    # ---- Final cross-experiment summary ----
    if len(configs) > 1:
        print(f"\n{'=' * 70}")
        print("CROSS-EXPERIMENT SUMMARY")
        print_summary_table(all_summaries)


if __name__ == "__main__":
    main()
