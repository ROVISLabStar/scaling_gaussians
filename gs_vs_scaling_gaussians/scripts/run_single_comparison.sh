#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nerfstudio

# Single-pair comparison: run all modes on one pair and plot convergence curves
# Run from mini_PVS_v2_github/gs_vs_scaling_gaussians/

INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
DATA_FACTOR=1
GS_MODEL="${INPUT}/output_gsplat_1_5_3_data_factor_${DATA_FACTOR}"
CKPT="${GS_MODEL}/ckpts/ckpt_29999_rank0.pt"
CFG="${GS_MODEL}/cfg.yml"

# Use the pair you showed to Guillaume (view 200→201, displaced 0.35m along Z)
START=200
GOAL=201
SF=1.8
OUT="logs/scale_vs/comparison"

# Run each mode sequentially in headless mode
for MODE in original inflated coarse_to_fine smooth_decay error_adaptive; do
    echo "=== Running ${MODE} ==="
    python experiments/scale_adaptive_vs.py \
        --ckpt "${CKPT}" --cfg "${CFG}" \
        --start_idx ${START} --goal_idx ${GOAL} \
        --displace_z 0.35 \
        --mode ${MODE} --scale_factor ${SF} \
        --data_factor 4 --max_iter 2000 \
        --out_dir "${OUT}/${MODE}" \
        --headless
done

# Plot all convergence curves together
cd ..
python -m gs_vs_scaling_gaussians.experiments.plot_results \
    --out_dir gs_vs_scaling_gaussians/figures/ \
    convergence \
    --files \
        "gs_vs_scaling_gaussians/${OUT}/original/convergence_original.npz" \
        "gs_vs_scaling_gaussians/${OUT}/inflated/convergence_inflated.npz" \
        "gs_vs_scaling_gaussians/${OUT}/coarse_to_fine/convergence_coarse_to_fine.npz" \
        "gs_vs_scaling_gaussians/${OUT}/smooth_decay/convergence_smooth_decay.npz" \
        "gs_vs_scaling_gaussians/${OUT}/error_adaptive/convergence_error_adaptive.npz"
