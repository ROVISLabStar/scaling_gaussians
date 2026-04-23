#!/bin/bash
# Launch viser comparison viewer for DDVS experiment results
# Usage: bash gs_vs_ddvs/scripts/launch_comparison_viewer.sh [LOG_DIR]
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nerfstudio

cd /home/youssefalj/Documents/SW/visual_navigation/mini_PVS_v2_github

LOG_DIR="${1:-gs_vs_ddvs/logs/test_run3}"

echo "============================================"
echo "DDVS Comparison Viewer"
echo "============================================"
echo "Log dir: ${LOG_DIR}"
echo "Open http://localhost:8080 in browser"
echo "============================================"

python -m gs_vs_ddvs.viser.comparison_viewer \
    --log_dir "${LOG_DIR}" \
    --port 8080
