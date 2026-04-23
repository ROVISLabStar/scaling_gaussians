#!/bin/bash


INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
DATA_FACTOR=1
GS_MODEL="${INPUT}/output_gsplat_1_5_3_data_factor_"${DATA_FACTOR}
NUM_ITERATIONS=6999 # 6999 or 29999



python -m gs_vs.experiments.servo_virtual_to_virtual_gsplat --ckpt ${GS_MODEL}/ckpts/ckpt_${NUM_ITERATIONS}_rank0.pt \
    --cfg  ${GS_MODEL}/cfg.yml \
    --camera_model pinhole \
    --feature_type pinhole \
    --desired_image_index 100
    
    
:<<"COMMENT"
python -m gs_vs.experiments.servo_virtual_to_virtual_gsplat --ckpt ${GS_MODEL}/ckpts/ckpt_${NUM_ITERATIONS}_rank0.pt \
    --cfg  ${GS_MODEL}/cfg.yml \
    --camera_model fisheye \
    --intrinsics_file config/intrinsics_fisheye_1024x1024.yml \
    --desired_image_index 100 \
    --feature_type unified_ps
COMMENT



