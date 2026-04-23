#!/bin/bash
# Batch evaluation: compare all modes across perturbation levels
# Run from mini_PVS_v2_github/

INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
DATA_FACTOR=1
GS_MODEL="${INPUT}/output_gsplat_1_5_3_data_factor_${DATA_FACTOR}"
CKPT="${GS_MODEL}/ckpts/ckpt_29999_rank0.pt"
CFG="${GS_MODEL}/cfg.yml"

python -m gs_vs_scaling_gaussians.experiments.run_scale_evaluation \
    --ckpt "${CKPT}" --cfg "${CFG}" \
    --data_factor 4 \
    --n_trials 20 \
    --levels small medium large \
    --modes original inflated coarse_to_fine smooth_decay error_adaptive \
    --scale_factor 1.8 \
    --max_iter 2000 \
    --seed 42 \
    --out_dir gs_vs_scaling_gaussians/logs/batch_evaluation

# Plot results
python -m gs_vs_scaling_gaussians.experiments.plot_results evaluation \
    --input gs_vs_scaling_gaussians/logs/batch_evaluation/evaluation_results.json \
    --out_dir gs_vs_scaling_gaussians/figures
