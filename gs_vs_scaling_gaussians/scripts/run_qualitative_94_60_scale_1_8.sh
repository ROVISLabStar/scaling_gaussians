#!/bin/bash
# Run all methods on pair 94->60, save every frame, generate videos
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nerfstudio

cd /home/youssefalj/Documents/SW/visual_navigation/mini_PVS_v2_github

INPUT="/media/youssefalj/LaCie/data/mip-nerf360/360_v2/room"
DATA_FACTOR=1
GS_MODEL="${INPUT}/output_gsplat_1_5_3_data_factor_${DATA_FACTOR}"
CKPT="${GS_MODEL}/ckpts/ckpt_29999_rank0.pt"
CFG="${GS_MODEL}/cfg.yml"

START=94
GOAL=60
MAX_ITER=2000
OUT_BASE="gs_vs_scaling_gaussians/logs/qualitative_${START}_${GOAL}"

# Run all methods saving every frame
echo "=== Running all methods on pair ${START}->${GOAL} ==="
python -m gs_vs_scaling_gaussians.experiments.render_comparison \
    --ckpt "${CKPT}" --cfg "${CFG}" \
    --start_idx ${START} --goal_idx ${GOAL} \
    --scale_factor 1.8 --data_factor 8 \
    --save_every 1 --max_iter ${MAX_ITER} \
    --out_dir "${OUT_BASE}"

# Generate videos
VIDEO_DIR="${OUT_BASE}/videos"
mkdir -p "${VIDEO_DIR}"

for MODE in original inflated pgm_vs; do
    FRAME_DIR="${OUT_BASE}/${MODE}"
    if [ -d "${FRAME_DIR}" ] && [ "$(ls ${FRAME_DIR}/frame_*.png 2>/dev/null | head -1)" ]; then
        N_FRAMES=$(ls ${FRAME_DIR}/frame_*.png | wc -l)
        echo "=== Generating video for ${MODE} (${N_FRAMES} frames) ==="
        ffmpeg -y -framerate 30 \
            -pattern_type glob -i "${FRAME_DIR}/frame_*.png" \
            -c:v libx264 -pix_fmt yuv420p \
            -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
            "${VIDEO_DIR}/${MODE}_${START}_${GOAL}.mp4" \
            2>/dev/null
        echo "  Saved ${VIDEO_DIR}/${MODE}_${START}_${GOAL}.mp4"
    else
        echo "  No frames for ${MODE}, skipping"
    fi
done

echo ""
echo "=== Done ==="
echo "Frames: ${OUT_BASE}/{original,inflated,pgm_vs}/"
echo "Videos: ${VIDEO_DIR}/"
