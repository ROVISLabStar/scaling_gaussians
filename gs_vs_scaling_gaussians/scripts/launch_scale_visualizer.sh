#!/bin/bash
# Launch the interactive scale visualizer
# Shows effect of α on 3D Gaussians + rendered images
# Usage: bash gs_vs_scaling_gaussians/scripts/launch_scale_visualizer.sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate nerfstudio

INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
DATA_FACTOR=1
GS_MODEL="${INPUT}/output_gsplat_1_5_3_data_factor_${DATA_FACTOR}"
CKPT="${GS_MODEL}/ckpts/ckpt_29999_rank0.pt"
CFG="${GS_MODEL}/cfg.yml"

cd /home/youssefalj/Documents/SW/visual_navigation/mini_PVS_v2_github

python -m gs_vs_scaling_gaussians.viser.scale_visualizer \
    --ckpt "${CKPT}" --cfg "${CFG}" \
    --data_factor 4 \
    --port 8080
