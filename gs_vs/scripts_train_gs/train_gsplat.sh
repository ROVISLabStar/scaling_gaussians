#!/bin/bash
set -e


INPUT="/home/youssefalj/Documents/data/playroom"
#INPUT="/home/youssefalj/Documents/data/aimovement"
#INPUT="/media/youssefalj/LaCie/data/zipnerf/nyc"
#INPUT="/home/youssefalj/Documents/data/tsukuba13"

OUTPUT="${INPUT}/output_gsplat_360"
DATA_FACTOR=1

: <<'END_COMMENT'
END_COMMENT


CUDA_VISIBLE_DEVICES=0 
    python ../gsplat/examples/simple_trainer.py mcmc --data_factor ${DATA_FACTOR} \
    --camera_model pinhole \
    --max_steps 30_000 \
    --data_dir "${INPUT}" \
    --result_dir "${OUTPUT}"\
    --disable_viewer


