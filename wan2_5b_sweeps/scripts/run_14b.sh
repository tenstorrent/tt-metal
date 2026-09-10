#!/usr/bin/env bash
set -o pipefail
cd /mnt/tt-data/teja/tt-metal
export TT_METAL_HOME=$PWD HF_HOME=/mnt/tt-data/teja/hf \
       TT_DIT_CACHE_DIR=/mnt/tt-data/teja/wan_cache \
       PYTHONPATH=$PWD:$(python3 -m site --user-site)
source python_env/bin/activate
for STEPS in 20 40; do
  echo "===== 14B 480p 81f steps=$STEPS ($(date)) ====="
  WAN14B_HEIGHT=480 WAN14B_WIDTH=832 WAN14B_FRAMES=81 WAN14B_STEPS=$STEPS WAN14B_TRACED=1 WAN14B_REPEAT=2 \
    python -m pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py \
    -k "inference_generate and bh_4x8_ring" -sv --timeout=5400 2>&1 | tee /home/ttuser/wan14b_480p_s${STEPS}.log
  echo "steps=$STEPS rc=${PIPESTATUS[0]}"
done
echo "=== 14B RUNS DONE $(date) ==="
echo "----- SUMMARY -----"; grep -hE "E2E_14B|CLIP_14B" /home/ttuser/wan14b_480p_s*.log
