#!/bin/bash



#GS_MODEL="/home/youssefalj/Documents/data/aimovement/output_gsplat_360"
GS_MODEL="/home/youssefalj/Documents/data/playroom/output_gsplat_360"
#GS_MODEL="/home/youssefalj/Documents/data/tsukuba13/output_gsplat_360"

NUM_ITERATIONS=6999 # 6999 or 29999




python -m gs_vs.experiments.render_equidistant --ckpt ${GS_MODEL}/ckpts/ckpt_${NUM_ITERATIONS}_rank0.pt \
    --cfg  ${GS_MODEL}/cfg.yml \
    --camera_model pinhole \
    --result_dir ${GS_MODEL}/pinhole_rendering_${NUM_ITERATIONS}
    

   
  :<<'COMMENT'   
python -m gs_vs.experiments.render_equidistant --ckpt ${GS_MODEL}/ckpts/ckpt_${NUM_ITERATIONS}_rank0.pt \
    --cfg  ${GS_MODEL}/cfg.yml \
    --camera_model fisheye \
    --result_dir ${GS_MODEL}/fisheye_rendering_1024_${NUM_ITERATIONS} \
    --intrinsics_file config/intrinsics_fisheye_1024x1024.yml


python -m gs_vs.experiments.render_equidistant --ckpt ${GS_MODEL}/ckpts/ckpt_${NUM_ITERATIONS}_rank0.pt \
    --cfg  ${GS_MODEL}/cfg.yml \
    --camera_model fisheye \
    --result_dir ${GS_MODEL}/fisheye_rendering_640_${NUM_ITERATIONS} \
    --intrinsics_file config/intrinsics_fisheye_640x512.yml
COMMENT
