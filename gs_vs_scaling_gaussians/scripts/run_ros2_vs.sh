#!/bin/bash
# Run UR10e Visual Servoing with 3DGS via ROS2
#
# This script launches 3 terminals:
# 1. UR10e driver with fake hardware + RViz
# 2. Activate velocity controller
# 3. VS node
#
# Usage:
#   bash gs_vs_scaling_gaussians/scripts/run_ros2_vs.sh

set -e

INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
DATA_FACTOR=1
GS_MODEL="${INPUT}/output_gsplat_1_5_3_data_factor_${DATA_FACTOR}"
CKPT="${GS_MODEL}/ckpts/ckpt_29999_rank0.pt"
CFG="${GS_MODEL}/cfg.yml"

GOAL_IDX=10
MODE="inflated"  # original, inflated
SCALE_FACTOR=1.8
GAIN=10.0

PROJECT_DIR="/home/youssefalj/Documents/SW/visual_navigation/mini_PVS_v2_github"
VS_NODE="${PROJECT_DIR}/gs_vs_scaling_gaussians/ros2/vs_node.py"

echo "============================================"
echo "UR10e Visual Servoing with 3DGS (ROS2)"
echo "============================================"
echo "Goal view: ${GOAL_IDX}"
echo "Mode: ${MODE}, scale: ${SCALE_FACTOR}"
echo ""

# Step 1: Launch UR10e with fake hardware
echo "[Step 1] Launch UR10e driver (fake hardware)..."
echo "Run in a NEW terminal:"
echo ""
echo "  source /opt/ros/humble/setup.bash"
echo "  ros2 launch ur_robot_driver ur_control.launch.py \\"
echo "      ur_type:=ur10e \\"
echo "      use_fake_hardware:=true \\"
echo "      launch_rviz:=true \\"
echo "      initial_joint_controller:=forward_velocity_controller"
echo ""
echo "Press ENTER when the driver is running..."
read

# Step 2: Check that the driver is running
echo "[Step 2] Checking joint states..."
source /opt/ros/humble/setup.bash
timeout 5 ros2 topic echo /joint_states --once 2>/dev/null | head -5
if [ $? -ne 0 ]; then
    echo "ERROR: No joint states received. Is the UR driver running?"
    exit 1
fi
echo "Joint states OK."
echo ""

# Step 3: Run VS node (use system Python 3.10 for ROS2 + gsplat compatibility)
echo "[Step 3] Starting VS node..."
source /opt/ros/humble/setup.bash
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

cd "${PROJECT_DIR}"
/usr/bin/python3 gs_vs_scaling_gaussians/ros2/vs_node.py --ros-args \
    -p ckpt:="${CKPT}" \
    -p cfg:="${CFG}" \
    -p data_factor:=8 \
    -p goal_idx:=${GOAL_IDX} \
    -p mode:="${MODE}" \
    -p scale_factor:=${SCALE_FACTOR} \
    -p gain:=${GAIN} \
    -p rate:=20.0
