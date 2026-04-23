#!/bin/bash
# Terminal 3: Launch VS node for UR10e visual servoing with 3DGS
# Usage: bash gs_vs_scaling_gaussians/scripts/launch_vs_node.sh [mode] [start] [goal] [scale] [gain] [threshold]
#
# Examples:
#   bash gs_vs_scaling_gaussians/scripts/launch_vs_node.sh inflated 12 10 1.8 10.0 0.1
#   bash gs_vs_scaling_gaussians/scripts/launch_vs_node.sh original 14 10 1.0 10.0 100
#   bash gs_vs_scaling_gaussians/scripts/launch_vs_node.sh pgm_vs 12 10 1.0 10.0 100

# Parameters (with defaults)
MODE="${1:-inflated}"
START_IDX="${2:-12}"
GOAL_IDX="${3:-10}"
SCALE_FACTOR="${4:-1.8}"
GAIN="${5:-10.0}"
CONV_THRESHOLD="${6:-10.0}"
MAX_ITER="${7:-2000}"

# Scene
INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
DATA_FACTOR=1
GS_MODEL="${INPUT}/output_gsplat_1_5_3_data_factor_${DATA_FACTOR}"
CKPT="${GS_MODEL}/ckpts/ckpt_29999_rank0.pt"
CFG="${GS_MODEL}/cfg.yml"

# Project
PROJECT_DIR="/home/youssefalj/Documents/SW/visual_navigation/mini_PVS_v2_github"

cd "${PROJECT_DIR}"
source /opt/ros/humble/setup.bash
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

echo "============================================"
echo "VS Node: ${MODE} mode"
echo "============================================"
echo "Pair: ${START_IDX} -> ${GOAL_IDX}"
echo "Scale factor: ${SCALE_FACTOR}"
echo "Gain: ${GAIN}"
echo "Conv. threshold: ${CONV_THRESHOLD}"
echo "Max iter: ${MAX_ITER}"
echo "============================================"
echo ""

/usr/bin/python3 gs_vs_scaling_gaussians/ros2/vs_node.py --ros-args \
    -p ckpt:="${CKPT}" \
    -p cfg:="${CFG}" \
    -p data_factor:=8 \
    -p goal_idx:=${GOAL_IDX} \
    -p start_idx:=${START_IDX} \
    -p mode:="${MODE}" \
    -p scale_factor:=${SCALE_FACTOR} \
    -p gain:=${GAIN} \
    -p mu:=0.01 \
    -p convergence_threshold:=${CONV_THRESHOLD} \
    -p max_iter:=${MAX_ITER} \
    -p rate:=10.0 \
    -p pgm_lambda_init:=5.0 \
    -p pgm_lambda_final:=1.0 \
    -p pgm_gain:=10.0
