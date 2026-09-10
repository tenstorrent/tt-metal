#!/usr/bin/env bash
# Wan2.2-14B (T2V-A14B) benchmark on single BH Galaxy (4x8): 480p + 720p, 20 & 40 steps.
set -o pipefail
cd /mnt/tt-data/teja/tt-metal
export TT_METAL_HOME=$PWD HF_HOME=/mnt/tt-data/teja/hf \
       TT_DIT_CACHE_DIR=/mnt/tt-data/teja/wan_cache \
       PYTHONPATH=$PWD:$(python3 -m site --user-site)
source python_env/bin/activate

run_one () {
  local H=$1 W=$2 STEPS=$3 TAG=$4
  echo "===== 14B ${TAG} ${W}x${H} 81f steps=${STEPS} ($(date)) ====="
  WAN14B_HEIGHT=$H WAN14B_WIDTH=$W WAN14B_FRAMES=81 WAN14B_STEPS=$STEPS \
    WAN14B_TRACED=1 WAN14B_REPEAT=2 WAN14B_SEED=42 \
    python -m pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py \
      -k "inference_generate and bh_4x8_ring" -sv --timeout=7200 \
      2>&1 | tee /home/ttuser/wan14b_${TAG}_s${STEPS}.log
  echo "${TAG} steps=${STEPS} rc=${PIPESTATUS[0]}"
}

# 720p first (most expensive; validates DRAM fits), then 480p. 40 then 20 steps.
run_one 720 1280 40 720p
run_one 720 1280 20 720p
run_one 480 832  40 480p
run_one 480 832  20 480p

echo "=== 14B ALL RUNS DONE $(date) ==="
echo "----- E2E / CLIP SUMMARY -----"
grep -hE "E2E_14B|CLIP_14B" /home/ttuser/wan14b_*_s*.log
