#!/usr/bin/env bash
set -o pipefail
cd /mnt/tt-data/teja/tt-metal
export TT_METAL_HOME=$PWD HF_HOME=/mnt/tt-data/teja/hf \
       TT_DIT_CACHE_DIR=/mnt/tt-data/teja/wan_cache PYTHONPATH=$PWD:$(python3 -m site --user-site)
# generation knobs (override before launching)
export WAN5B_FRAMES=${WAN5B_FRAMES:-81}
export WAN5B_STEPS=${WAN5B_STEPS:-30}
export WAN5B_HEIGHT=${WAN5B_HEIGHT:-704}
export WAN5B_WIDTH=${WAN5B_WIDTH:-1280}
export WAN5B_SEED=${WAN5B_SEED:-42}
export WAN5B_FPS=${WAN5B_FPS:-24}
export WAN5B_TRACED=${WAN5B_TRACED:-0}
export WAN5B_REPEAT=${WAN5B_REPEAT:-1}
echo "=== GENERATE START $(date) frames=$WAN5B_FRAMES steps=$WAN5B_STEPS traced=$WAN5B_TRACED repeat=$WAN5B_REPEAT fps=$WAN5B_FPS ${WAN5B_HEIGHT}x${WAN5B_WIDTH} seed=$WAN5B_SEED ==="
source python_env/bin/activate
python -m pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_ti2v_5b.py \
  -k "generate and bh_4x8" -sv --timeout=7200
echo "=== GENERATE EXIT=$? $(date) ==="
