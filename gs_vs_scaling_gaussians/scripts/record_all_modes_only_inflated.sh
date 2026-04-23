#!/bin/bash
# Record all VS modes into a single rosbag
# Usage: bash gs_vs_scaling_gaussians/scripts/record_all_modes.sh [start] [goal]

# for room sequence this couple shows that inflated only converges.
#START_IDX="${1:-19}"
#GOAL_IDX="${2:-149}"

START_IDX="${1:-19}"
GOAL_IDX="${2:-149}"

BAG_DIR="gs_vs_scaling_gaussians/logs/rosbag_${START_IDX}_${GOAL_IDX}"

PROJECT_DIR="/home/youssefalj/Documents/SW/visual_navigation/mini_PVS_v2_github"
INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
CKPT="${INPUT}/output_gsplat_1_5_3_data_factor_1/ckpts/ckpt_29999_rank0.pt"
CFG="${INPUT}/output_gsplat_1_5_3_data_factor_1/cfg.yml"

cd "${PROJECT_DIR}"
source /opt/ros/humble/setup.bash
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

mkdir -p "${BAG_DIR}"

# Kill everything first
killall -9 ros2_control_node urscript_interface robot_state_publisher python3 2>/dev/null
sleep 2

echo "============================================"
echo "Recording all VS modes: ${START_IDX} -> ${GOAL_IDX}"
echo "============================================"

# Step 1: Launch UR driver
echo "Starting UR driver..."
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur10e robot_ip:=0.0.0.0 \
    use_fake_hardware:=true launch_rviz:=false \
    initial_joint_controller:=forward_velocity_controller \
    > /dev/null 2>&1 &

echo -n "Waiting for driver"
for i in $(seq 1 20); do
    timeout 1 ros2 topic echo /joint_states --once > /dev/null 2>&1 && break
    echo -n "."
    sleep 1
done
echo " ready!"
pkill -f "urscript_interface" 2>/dev/null

# Step 2: Start recording (stays running for all modes)
rm -rf "${BAG_DIR}/vs_all_modes" 2>/dev/null
ros2 bag record -o "${BAG_DIR}/vs_all_modes" \
    /joint_states /tf \
    /vs/current_image /vs/desired_image /vs/diff_image \
    /vs/scene_pointcloud \
    /vs/photometric_error /vs/pose_error_t /vs/pose_error_r \
    /forward_velocity_controller/commands \
    > /dev/null 2>&1 &
BAG_PID=$!
echo "Recording started (PID: ${BAG_PID})"

# Step 3: Run each mode
run_vs() {
    local MODE=$1
    local SCALE=$2
    local GAIN=$3
    local THRESH=$4

    echo ""
    echo "====== ${MODE} (scale=${SCALE}, thresh=${THRESH}) ======"
    echo "  Loading scene (takes ~60s first time)..."

    timeout 600 /usr/bin/python3 gs_vs_scaling_gaussians/ros2/vs_node.py --ros-args \
        -p ckpt:="${CKPT}" \
        -p cfg:="${CFG}" \
        -p data_factor:=8 \
        -p goal_idx:=${GOAL_IDX} \
        -p start_idx:=${START_IDX} \
        -p mode:="${MODE}" \
        -p scale_factor:=${SCALE} \
        -p gain:=${GAIN} \
        -p mu:=0.01 \
        -p convergence_threshold:=${THRESH} \
        -p rate:=10.0 \
        -p pgm_lambda_init:=5.0 \
        -p pgm_lambda_final:=1.0 \
        -p pgm_gain:=10.0 \
    2>&1 | grep -E "\[INFO\].*vs_node"

    echo "  ====== ${MODE} done ======"
    sleep 2
}

run_vs "original"  1.0  10.0  3.0
run_vs "inflated"  2.5  10.0  0.1
run_vs "pgm_vs"    1.0  10.0  3.0

# Step 4: Stop recording
echo ""
echo "Stopping recording..."
kill ${BAG_PID} 2>/dev/null
wait ${BAG_PID} 2>/dev/null

# Cleanup
pkill -9 -f "ros2_control_node|urscript|robot_state_publisher|trajectory_until" 2>/dev/null

echo ""
echo "============================================"
echo "Done! Checking bag:"
ros2 bag info "${BAG_DIR}/vs_all_modes" 2>&1 | grep -E "Duration|Messages|Count"
echo ""
echo "Replay: ros2 bag play ${BAG_DIR}/vs_all_modes --loop"
echo "View:   bash gs_vs_scaling_gaussians/scripts/launch_rqt.sh"
echo "============================================"
