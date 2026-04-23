#!/bin/bash
# Scale factor sweep: find optimal scale factor for medium perturbations
# Run from mini_PVS_v2_github/

INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
DATA_FACTOR=1
GS_MODEL="${INPUT}/output_gsplat_1_5_3_data_factor_${DATA_FACTOR}"
CKPT="${GS_MODEL}/ckpts/ckpt_29999_rank0.pt"
CFG="${GS_MODEL}/cfg.yml"

python -m gs_vs_scaling_gaussians.experiments.sweep_scale_factor \
    --ckpt "${CKPT}" --cfg "${CFG}" \
    --data_factor 4 \
    --n_trials 15 \
    --level medium \
    --scale_min 1.0 --scale_max 5.0 --scale_steps 9 \
    --max_iter 2000 \
    --seed 42 \
    --out_dir gs_vs_scaling_gaussians/logs/scale_sweep

# Plot
python -m gs_vs_scaling_gaussians.experiments.plot_results sweep \
    --input gs_vs_scaling_gaussians/logs/scale_sweep/sweep_results.json \
    --out_dir gs_vs_scaling_gaussians/figures
