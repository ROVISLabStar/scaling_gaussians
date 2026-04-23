#!/bin/bash
# Terminal 2: Launch RViz on NVIDIA GPU with throttled TF
# Usage: bash gs_vs_scaling_gaussians/scripts/launch_rviz.sh

source /opt/ros/humble/setup.bash

RVIZ_CONFIG="/home/youssefalj/Documents/SW/visual_navigation/mini_PVS_v2_github/gs_vs_scaling_gaussians/ros2/config/ur10e.rviz"

# Throttle TF to 10 Hz (UR driver publishes at 500 Hz which overwhelms RViz)
ros2 run topic_tools throttle messages /tf 10.0 /tf_throttled &
THROTTLE_PID=$!

echo "TF throttled to 10 Hz (PID: ${THROTTLE_PID})"
echo "Launching RViz2 on NVIDIA GPU..."

# Launch RViz
exec env __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
    rviz2 -d "${RVIZ_CONFIG}"
