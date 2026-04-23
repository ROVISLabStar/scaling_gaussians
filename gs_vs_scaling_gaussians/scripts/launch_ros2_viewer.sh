#!/bin/bash
# Launch ROS2-connected Viser viewer (3DGS scene + drone)
# Generic — no start/goal needed. Just shows whatever the VS node publishes.
# Usage: bash gs_vs_scaling_gaussians/scripts/launch_ros2_viewer.sh

PROJECT_DIR="/home/youssefalj/Documents/SW/visual_navigation/mini_PVS_v2_github"
INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
CKPT="${INPUT}/output_gsplat_1_5_3_data_factor_1/ckpts/ckpt_29999_rank0.pt"
CFG="${INPUT}/output_gsplat_1_5_3_data_factor_1/cfg.yml"

cd "${PROJECT_DIR}"
source /opt/ros/humble/setup.bash
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

echo "============================================"
echo "ROS2 Viser Viewer (3DGS + drone)"
echo "============================================"
echo "Open http://localhost:8080 in browser"
echo "Then start VS node in another terminal"
echo "============================================"

/usr/bin/python3 gs_vs_scaling_gaussians/viser/ros2_viewer.py \
    --ckpt "${CKPT}" --cfg "${CFG}" \
    --data_factor 8 \
    --port 8080
