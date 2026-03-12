#!/bin/bash



#GS_MODEL="/home/youssefalj/Documents/data/aimovement/output_gsplat_360"
GS_MODEL="/home/youssefalj/Documents/data/playroom/output_gsplat_360"
#GS_MODEL="/home/youssefalj/Documents/data/tsukuba13/output_gsplat_360"

NUM_ITERATIONS=29999 # 29999 or 6999

:<<"COMMENT"



python -m gs_vs.experiments.servo_virtual_to_virtual_gsplat --ckpt ${GS_MODEL}/ckpts/ckpt_${NUM_ITERATIONS}_rank0.pt \
    --cfg  ${GS_MODEL}/cfg.yml \
    --camera_model fisheye \
    --intrinsics_file config/intrinsics_fisheye_1024x1024.yml \
    --desired_image_index 100 \
    --feature_type unified_ps
COMMENT

python -m gs_vs.experiments.servo_virtual_to_virtual_gsplat --ckpt ${GS_MODEL}/ckpts/ckpt_${NUM_ITERATIONS}_rank0.pt \
    --cfg  ${GS_MODEL}/cfg.yml \
    --camera_model pinhole \
    --feature_type pinhole \
    --desired_image_index 100

