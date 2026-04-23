#!/bin/bash
# Launch the interactive 3D viewer for UR10 VS with 3DGS
# Open http://localhost:8080 in your browser after launching
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nerfstudio

INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
DATA_FACTOR=1
GS_MODEL="${INPUT}/output_gsplat_1_5_3_data_factor_${DATA_FACTOR}"
CKPT="${GS_MODEL}/ckpts/ckpt_29999_rank0.pt"
CFG="${GS_MODEL}/cfg.yml"

cd /home/youssefalj/Documents/SW/visual_navigation/mini_PVS_v2_github

python -m gs_vs_scaling_gaussians.viser.interactive_viewer \
    --ckpt "${CKPT}" --cfg "${CFG}" \
    --data_factor 4 \
    --port 8080
