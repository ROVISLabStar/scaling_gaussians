INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
DATA_FACTOR=1
GS_MODEL="${INPUT}/output_gsplat_1_5_3_data_factor_${DATA_FACTOR}"
CKPT="${GS_MODEL}/ckpts/ckpt_29999_rank0.pt"
CFG="${GS_MODEL}/cfg.yml"

#--start_idx 255 --goal_idx 72 \

# Test on the hard pair
python experiments/scale_adaptive_vs.py \
    --ckpt "${CKPT}" --cfg "${CFG}" \
    --start_idx 255 --goal_idx 72 \
    --mode inflated --scale_factor 1.8 --data_factor 2 \
    --max_iter 2000 --port 8080 \
    --out_dir logs/scale_vs/hard_inflated_1.8
    
    
:<<"comment"
# 1. Original scales — should PLATEAU (we already know this)
python experiments/scale_adaptive_vs.py \
    --ckpt "${CKPT}" --cfg "${CFG}" \
    --start_idx 255 --goal_idx 253 \
    --mode original --data_factor 4 \
    --max_iter 2000 --port 8080 \
    --out_dir logs/scale_vs/original


# Mild inflation ×1.5
python experiments/scale_adaptive_vs.py \
    --ckpt "${CKPT}" --cfg "${CFG}" \
    --start_idx 255 --goal_idx 253 \
    --mode inflated --scale_factor 1.80 --data_factor 4 \
    --max_iter 2000 --port 8081 \
    --out_dir logs/scale_vs/inflated_1.5
comment



:<<"comment"    
python experiments/scale_adaptive_vs.py \
    --ckpt "${CKPT}" --cfg "${CFG}" \
    --start_idx 255 --goal_idx 253 \
    --mode smooth_decay --scale_factor 1.80 --data_factor 4 \
    --max_iter 2000 --port 8081 \
    --out_dir logs/scale_vs/inflated_1.5
comment








    
:<<"comment"
# Smooth decay from 1.5 to 1.0
python experiments/scale_adaptive_vs.py \
    --ckpt "${CKPT}" --cfg "${CFG}" \
    --start_idx 255 --goal_idx 72 \
    --mode smooth_decay --scale_factor 1.5 --data_factor 4 \
    --max_iter 2000 --port 8081 \
    --out_dir logs/scale_vs/smooth_1.5
comment
