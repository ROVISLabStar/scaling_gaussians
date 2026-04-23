#!/bin/bash
# Train gsplat on multiple mip-nerf360 scenes for the TRO paper
# Run from mini_PVS_v2_github/
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate nerfstudio

SCENES="garden kitchen bicycle"
DATA_FACTOR=1
BASE="/media/youssefalj/LaCie/data/mip-nerf360/360_v2"

for SCENE in ${SCENES}; do
    INPUT="${BASE}/${SCENE}"
    GS_MODEL="${INPUT}/output_gsplat_1_5_3_data_factor_${DATA_FACTOR}"

    if [ -f "${GS_MODEL}/ckpts/ckpt_29999_rank0.pt" ]; then
        echo "=== ${SCENE}: already trained, skipping ==="
        continue
    fi

    echo "=== Training ${SCENE} ==="
    echo "  Input: ${INPUT}"
    echo "  Output: ${GS_MODEL}"

    python gsplat/examples/simple_trainer.py mcmc \
        --data_factor ${DATA_FACTOR} \
        --camera_model pinhole \
        --max_steps 30_000 \
        --data_dir "${INPUT}" \
        --result_dir "${GS_MODEL}" \
        --disable_viewer

    echo "=== ${SCENE}: done ==="
    echo ""
done

echo "All scenes trained."
