python experiments/compare_perspective_vs_spherical.py \
    --ckpt /media/youssefalj/LaCie/data/mip-nerf360/360_v2/room/output_gsplat_1_5_3_data_factor_1/ckpts/ckpt_29999_rank0.pt \
    --cfg /media/youssefalj/LaCie/data/mip-nerf360/360_v2/room/output_gsplat_1_5_3_data_factor_1/cfg.yml \
    --feature_type pinhole \
    --desired_image_index 0 \
    --delta_t 0.05 0.05 -0.05 \
    --delta_r 0 5 0 \
    --port 8080
