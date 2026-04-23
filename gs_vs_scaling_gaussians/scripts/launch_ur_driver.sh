#!/bin/bash
# Terminal 1: Launch UR10e driver (fake hardware) with reduced update rate
# Usage: bash gs_vs_scaling_gaussians/scripts/launch_ur_driver.sh

source /opt/ros/humble/setup.bash

# Kill any leftover ROS processes
ps aux | grep -E "ros2_control|urscript|robot_state|controller_manager|rviz" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
sleep 2

# Temporarily reduce update rate (backup and replace)
UR_CONFIG="/opt/ros/humble/share/ur_robot_driver/config/ur10e_update_rate.yaml"
BACKUP="/tmp/ur10e_update_rate_backup.yaml"

if [ ! -f "${BACKUP}" ]; then
    sudo cp "${UR_CONFIG}" "${BACKUP}"
fi
echo "controller_manager:" | sudo tee "${UR_CONFIG}" > /dev/null
echo "  ros__parameters:" | sudo tee -a "${UR_CONFIG}" > /dev/null
echo "    update_rate: 50" | sudo tee -a "${UR_CONFIG}" > /dev/null
echo "Update rate set to 50 Hz"

echo "============================================"
echo "UR10e Driver (fake hardware, 50 Hz)"
echo "============================================"
echo ""
echo "After this starts, open TWO more terminals:"
echo ""
echo "  Terminal 2 (RViz):"
echo "    bash gs_vs_scaling_gaussians/scripts/launch_rviz.sh"
echo ""
echo "  Terminal 3 (VS node):"
echo "    bash gs_vs_scaling_gaussians/scripts/launch_vs_node.sh inflated 12 10 1.8 5.0"
echo "============================================"
echo ""

# Launch in background, kill urscript after startup, then wait
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur10e \
    robot_ip:=0.0.0.0 \
    use_fake_hardware:=true \
    launch_rviz:=false \
    initial_joint_controller:=forward_velocity_controller \
    &
LAUNCH_PID=$!
sleep 10
pkill -f "urscript_interface" 2>/dev/null
pkill -f "trajectory_until" 2>/dev/null
echo "Killed urscript_interface (no more connection errors)"
wait $LAUNCH_PID

# Restore original on exit
if [ -f "${BACKUP}" ]; then
    sudo cp "${BACKUP}" "${UR_CONFIG}"
    echo "Update rate restored to 500 Hz"
fi
