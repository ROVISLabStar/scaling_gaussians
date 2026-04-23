INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
DATA_FACTOR=1
GS_MODEL="${INPUT}/output_gsplat_1_5_3_data_factor_${DATA_FACTOR}"
CKPT="${GS_MODEL}/ckpts/ckpt_29999_rank0.pt"
CFG="${GS_MODEL}/cfg.yml"

python experiments/visualize_scales.py \
    --ckpt "${CKPT}" --cfg "${CFG}" \
    --view_idx 128 --data_factor 2 \
    --out_dir logs/scale_viz
