INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
DATA_FACTOR=1
GS_MODEL="${INPUT}/output_gsplat_1_5_3_data_factor_${DATA_FACTOR}"
CKPT="${GS_MODEL}/ckpts/ckpt_29999_rank0.pt"
CFG="${GS_MODEL}/cfg.yml"

# Original scale, 0.1m backward
python experiments/scale_adaptive_vs.py \
    --ckpt "${CKPT}" --cfg "${CFG}" \
    --start_idx 200 --goal_idx 201 \
    --displace_z 0.35 \
    --mode original --data_factor 4 \
    --max_iter 2000 --port 8080 \
    --out_dir logs/scale_vs/tz01_original

# Inflated ×1.8, same displacement
python experiments/scale_adaptive_vs.py \
    --ckpt "${CKPT}" --cfg "${CFG}" \
    --start_idx 200 --goal_idx 201 \
    --displace_z 0.35 \
    --mode inflated --scale_factor 1.8 --data_factor 4 \
    --max_iter 2000 --port 8081 \
    --out_dir logs/scale_vs/tz01_inflated
