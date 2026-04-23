#!/bin/bash
# Run DDVS comparison: Original vs DDVS vs Inflated 3DGS vs PGM-VS
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nerfstudio

cd /home/youssefalj/Documents/SW/visual_navigation/mini_PVS_v2_github

INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
CKPT="${INPUT}/output_gsplat_1_5_3_data_factor_1/ckpts/ckpt_29999_rank0.pt"
CFG="${INPUT}/output_gsplat_1_5_3_data_factor_1/cfg.yml"
START=19
GOAL=149

python -m gs_vs_ddvs.experiments.ddvs_servo \
    --ckpt "${CKPT}" --cfg "${CFG}" \
    --start_idx ${START} --goal_idx ${GOAL} \
    --aperture_phi 2.0 --focus_depth 0.5 \
    --scale_factor 2.5 \
    --data_factor 8 \
    --max_iter 2000 \
    --save_every 20 \
    --out_dir gs_vs_ddvs/logs/ddvs_${START}_${GOAL}
