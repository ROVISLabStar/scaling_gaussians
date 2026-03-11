#!/bin/bash



#GS_MODEL="/home/youssefalj/Documents/data/aimovement/output_gsplat_360"
#GS_MODEL="/home/youssefalj/Documents/data/playroom/output_gsplat_360/"
GS_MODEL="/home/youssefalj/Documents/data/tsukuba13/output_gsplat_360"
DESIRED_IMAGE="/home/youssefalj/Documents/SW/visual_navigation/minimal_pvs_virtual/desired_images/fisheye_000000.png"


python -m gs_vs.experiments.servo_virtual_to_real_gsplat --ckpt ${GS_MODEL}/ckpts/ckpt_${NUM_ITERATIONS}_rank0.pt \
    --cfg  ${GS_MODEL}/cfg.yml \
    --camera_model fisheye \
    --intrinsics_file config/intrinsics_fisheye_1024x1024.yml \
    --desired_image ${DESIRED_IMAGE} \
    --feature_type unified_cs

    
    
python ./examples/servo_virtual_to_real_gsplat.py --ckpt ${GS_MODEL}/ckpts/ckpt_6999_rank0.pt \
    --cfg  ${GS_MODEL}/cfg.yml \
    --camera_model fisheye \
    --desired_image ${DESIRED_IMAGE}

 
 
