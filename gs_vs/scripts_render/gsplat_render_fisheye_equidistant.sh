#!/bin/bash



INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
DATA_FACTOR=1
GS_MODEL="${INPUT}/output_gsplat_1_5_3_data_factor_"${DATA_FACTOR}
NUM_ITERATIONS=29999 # 6999 or 29999
RENDERING_DIR="${GS_MODEL}/fisheye_rendering_${NUM_ITERATIONS}"

python -m gs_vs.experiments.render --ckpt ${GS_MODEL}/ckpts/ckpt_${NUM_ITERATIONS}_rank0.pt \
    --cfg  ${GS_MODEL}/cfg.yml \
    --camera_model fisheye \
    --result_dir ${RENDERING_DIR}
    --intrinsics_file config/intrinsics_fisheye_1024x1024.yml



:<<'COMMENT'   
python -m gs_vs.experiments.render --ckpt ${GS_MODEL}/ckpts/ckpt_${NUM_ITERATIONS}_rank0.pt \
    --cfg  ${GS_MODEL}/cfg.yml \
    --camera_model fisheye \
    --result_dir ${RENDERING_DIR} \
    --intrinsics_file config/intrinsics_fisheye_640x512.yml
COMMENT
