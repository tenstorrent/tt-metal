#!/usr/bin/env bash
set -o pipefail
cd /mnt/tt-data/teja/tt-metal
export TT_METAL_HOME=$PWD HF_HOME=/mnt/tt-data/teja/hf \
       TT_DIT_CACHE_DIR=/mnt/tt-data/teja/wan_cache \
       PYTHONPATH=$PWD:$(python3 -m site --user-site)
source python_env/bin/activate
run() {
  local W=$1 H=$2 TAG=$3
  echo "===== 5B ${TAG} ${W}x${H} 121f/40steps traced ($(date)) ====="
  WAN5B_WIDTH=$W WAN5B_HEIGHT=$H WAN5B_FRAMES=81 WAN5B_STEPS=40 \
  WAN5B_TRACED=1 WAN5B_REPEAT=2 WAN5B_CLIP=0 \
  WAN5B_OUT=/home/ttuser/wan5b40_${TAG}_${W}x${H}.mp4 \
    python -m pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_ti2v_5b.py \
    -k "generate and bh_4x8" -sv --timeout=5400 2>&1 | tee /home/ttuser/wan5b40_${TAG}.log
  echo "${TAG} rc=${PIPESTATUS[0]}"
}
run 832 480 480p
run 1280 704 720p
echo "=== 5B 40-STEP RUNS DONE $(date) ==="
echo "----- SUMMARY -----"
grep -hE "E2E_PIPELINE_TIME \[warm_traced\]|HOST_VS_DEVICE \[warm_traced\]" /home/ttuser/wan5b40_480p.log /home/ttuser/wan5b40_720p.log
