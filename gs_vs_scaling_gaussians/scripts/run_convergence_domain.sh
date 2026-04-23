#!/bin/bash
# Convergence domain estimation: find max displacement per axis per mode
# Run from mini_PVS_v2_github/

INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
DATA_FACTOR=1
GS_MODEL="${INPUT}/output_gsplat_1_5_3_data_factor_${DATA_FACTOR}"
CKPT="${GS_MODEL}/ckpts/ckpt_29999_rank0.pt"
CFG="${GS_MODEL}/cfg.yml"

python -m gs_vs_scaling_gaussians.experiments.estimate_convergence_domain \
    --ckpt "${CKPT}" --cfg "${CFG}" \
    --data_factor 4 \
    --modes original inflated coarse_to_fine error_adaptive \
    --scale_factors 1.0 1.5 2.0 3.0 \
    --axes tx ty tz rx ry rz \
    --n_bisect 8 --n_confirm 1 \
    --t_max 0.5 --r_max 40.0 \
    --max_iter 2000 \
    --out_dir gs_vs_scaling_gaussians/logs/convergence_domain

# Plot
python -m gs_vs_scaling_gaussians.experiments.plot_results domain \
    --input gs_vs_scaling_gaussians/logs/convergence_domain/convergence_domain.json \
    --out_dir gs_vs_scaling_gaussians/figures
