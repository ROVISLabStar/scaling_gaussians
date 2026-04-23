#!/bin/bash
# Run UR10e VS with ROS2 and record rosbag
# No RViz needed — visualization via viser or offline playback
source /opt/ros/humble/setup.bash

# Kill any existing ROS processes
ps aux | grep -E "ros2_control|urscript|robot_state|controller|spawner" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
sleep 3

INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
CKPT="${INPUT}/output_gsplat_1_5_3_data_factor_1/ckpts/ckpt_29999_rank0.pt"
CFG="${INPUT}/output_gsplat_1_5_3_data_factor_1/cfg.yml"
PROJECT_DIR="/home/youssefalj/Documents/SW/visual_navigation/mini_PVS_v2_github"
BAG_DIR="${PROJECT_DIR}/gs_vs_scaling_gaussians/logs/rosbags"

GOAL_IDX=${1:-10}
START_IDX=${2:-12}
MODE=${3:-inflated}
SCALE=${4:-1.8}
GAIN=${5:-0.5}

echo "============================================"
echo "UR10e VS with 3DGS (ROS2 + rosbag)"
echo "============================================"
echo "Pair: ${START_IDX} -> ${GOAL_IDX}"
echo "Mode: ${MODE}, scale: ${SCALE}, gain: ${GAIN}"
echo ""

# Step 1: Launch UR driver (background)
echo "[1] Launching UR10e driver..."
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur10e robot_ip:=0.0.0.0 \
    use_fake_hardware:=true launch_rviz:=false \
    initial_joint_controller:=forward_velocity_controller \
2>/dev/null &
UR_PID=$!
sleep 12

# Kill urscript to avoid error spam
pkill -f urscript_interface 2>/dev/null
pkill -f trajectory_until 2>/dev/null

# Verify
if ! ros2 topic list 2>/dev/null | grep -q joint_states; then
    echo "ERROR: UR driver failed to start"
    kill $UR_PID 2>/dev/null
    exit 1
fi
echo "[1] UR driver running."

# Step 2: Record rosbag (background)
echo "[2] Recording rosbag..."
mkdir -p "${BAG_DIR}"
BAG_NAME="vs_${MODE}_${START_IDX}_${GOAL_IDX}_$(date +%Y%m%d_%H%M%S)"
ros2 bag record \
    /joint_states \
    /tf \
    /vs/current_image \
    /vs/desired_image \
    /vs/diff_image \
    /forward_velocity_controller/commands \
    -o "${BAG_DIR}/${BAG_NAME}" \
2>/dev/null &
BAG_PID=$!
echo "[2] Recording to ${BAG_DIR}/${BAG_NAME}"

# Step 3: Run VS node
echo "[3] Starting VS node..."
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
cd "${PROJECT_DIR}"

/usr/bin/python3 gs_vs_scaling_gaussians/ros2/vs_node.py --ros-args \
    -p ckpt:="${CKPT}" \
    -p cfg:="${CFG}" \
    -p data_factor:=8 \
    -p goal_idx:=${GOAL_IDX} \
    -p start_idx:=${START_IDX} \
    -p mode:="${MODE}" \
    -p scale_factor:=${SCALE} \
    -p gain:=${GAIN} \
    -p rate:=10.0

# Cleanup
echo ""
echo "[4] Stopping recording and driver..."
kill $BAG_PID 2>/dev/null
kill $UR_PID 2>/dev/null
pkill -f "ros2_control\|robot_state" 2>/dev/null

echo "Done."
echo "Rosbag saved to: ${BAG_DIR}/${BAG_NAME}"
echo ""
echo "To replay: ros2 bag play ${BAG_DIR}/${BAG_NAME}"
