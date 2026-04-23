#!/bin/bash

# ============================================================
# Visual Servoing on Gaussian Splatting
# Supports: --mode pgm | pl | both
# ============================================================

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate nerfstudio

# --- Mode: pgm, pl, or both ---
MODE="pl"  # "pgm" = PGM-VS only, "pl" = PL-VS only, "both" = comparison

# --- GS model info ---
INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
DATA_FACTOR=1
GS_MODEL="${INPUT}/output_gsplat_1_5_3_data_factor_${DATA_FACTOR}"
NUM_ITERATIONS=29999 # 6999 or 29999
CKPT="${GS_MODEL}/ckpts/ckpt_${NUM_ITERATIONS}_rank0.pt"
CFG="${GS_MODEL}/cfg.yml"

# --- PGM-VS parameters ---
LAMBDA_GI=25.0          # Initial extension parameter (large -> wide convergence)
LAMBDA_G_FINAL=1.0      # Final extension parameter (small -> high accuracy)
SWITCH_THRESHOLD=0.1    # Step 1 -> Step 2 transition threshold
GAIN_PGM=1.0            # PGM control gain

# --- PL-VS parameters ---
FEATURE_TYPE_PL="pinhole"  # pinhole, unified_ip, unified_cs, unified_ps, equidistant
MU_PL=0.01                 # LM damping
LAMBDA_PL=60.0             # PL control gain

# --- Common parameters ---
MAX_ITER=2000
CONVERGENCE_THRESHOLD=10  # MSE threshold (||e||^2 / n_pixels)
DESIRED_IMAGE_INDEX=0
CAMERA_MODEL="pinhole"
DISPLACEMENT="0.2,0.2,-0.2,0,0,0"  # tx,ty,tz,rx,ry,rz (meters,degrees)

cd "$(dirname "$0")/.."

python experiments/pgm_vs.py \
    --mode ${MODE} \
    --ckpt ${CKPT} \
    --cfg ${CFG} \
    --camera_model ${CAMERA_MODEL} \
    --desired_image_index ${DESIRED_IMAGE_INDEX} \
    --lambda_gi ${LAMBDA_GI} \
    --lambda_g_final ${LAMBDA_G_FINAL} \
    --switch_threshold ${SWITCH_THRESHOLD} \
    --gain_pgm ${GAIN_PGM} \
    --feature_type_pl ${FEATURE_TYPE_PL} \
    --mu_pl ${MU_PL} \
    --lambda_pl ${LAMBDA_PL} \
    --max_iter ${MAX_ITER} \
    --convergence_threshold ${CONVERGENCE_THRESHOLD} \
    --displacement ${DISPLACEMENT}
