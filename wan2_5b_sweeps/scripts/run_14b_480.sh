#!/usr/bin/env bash
set -o pipefail
cd /mnt/tt-data/teja/tt-metal
export TT_METAL_HOME=$PWD HF_HOME=/mnt/tt-data/teja/hf \
       TT_DIT_CACHE_DIR=/mnt/tt-data/teja/wan_cache \
       PYTHONPATH=$PWD:$(python3 -m site --user-site)
source python_env/bin/activate
echo "===== 14B 480p 832x480 81f steps=40 ($(date)) ====="
WAN14B_HEIGHT=480 WAN14B_WIDTH=832 WAN14B_FRAMES=81 WAN14B_STEPS=40 \
  WAN14B_TRACED=1 WAN14B_REPEAT=2 WAN14B_SEED=42 \
  python -m pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py \
    -k "inference_generate and bh_4x8_ring" -sv --timeout=7200 \
    2>&1 | tee /home/ttuser/wan14b_480p_s40.log
echo "480p steps=40 rc=${PIPESTATUS[0]}"
echo "=== 14B 480p DONE $(date) ==="
