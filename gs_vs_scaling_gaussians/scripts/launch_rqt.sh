#!/bin/bash
# Launch rqt with VS visualization on NVIDIA GPU
# Usage: bash gs_vs_scaling_gaussians/scripts/launch_rqt.sh

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
source /opt/ros/humble/setup.bash

PROJECT_DIR="/home/youssefalj/Documents/SW/visual_navigation/mini_PVS_v2_github"
PERSPECTIVE="${PROJECT_DIR}/gs_vs_scaling_gaussians/ros2/config/vs_layout.perspective"

if [ -f "${PERSPECTIVE}" ]; then
    echo "Loading saved perspective..."
    rqt --perspective-file "${PERSPECTIVE}"
else
    echo "No saved perspective found."
    echo ""
    echo "Configure rqt manually:"
    echo "  1. Plugins → Visualization → Image View (×3)"
    echo "     Topics: /vs/current_image, /vs/desired_image, /vs/diff_image"
    echo "  2. Plugins → Visualization → Plot"
    echo "     Topic: /vs/photometric_error"
    echo "  3. Arrange panels by dragging"
    echo "  4. File → Save Perspective As → save to:"
    echo "     ${PERSPECTIVE}"
    echo ""
    rqt
fi
