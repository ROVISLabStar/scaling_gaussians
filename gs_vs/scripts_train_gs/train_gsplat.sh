#!/bin/bash
set -e


INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
DATA_FACTOR=1
GS_MODEL="${INPUT}/output_gsplat_1_5_3_data_factor_"${DATA_FACTOR}
: <<'END_COMMENT'
END_COMMENT


CUDA_VISIBLE_DEVICES=0 
    python ../gsplat/examples/simple_trainer.py mcmc --data_factor ${DATA_FACTOR} \
    --camera_model pinhole \
    --max_steps 30_000 \
    --data_dir "${INPUT}" \
    --result_dir "${GS_MODEL}"\
    --disable_viewer


