#!/bin/bash
# Kill all ROS2 and VS processes
# Usage: bash gs_vs_scaling_gaussians/scripts/kill_all.sh

killall -9 ros2_control_node urscript_interface robot_state_publisher \
    rviz2 rqt python3 ros2 spawner trajectory_until_node 2>/dev/null
sleep 2
echo "All processes killed."
