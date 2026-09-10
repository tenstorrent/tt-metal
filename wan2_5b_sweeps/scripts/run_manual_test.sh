#!/usr/bin/env bash
set +e
ROOT=/mnt/tt-data/teja/tt-metal
export TT_METAL_HOME=$ROOT HF_HOME=/mnt/tt-data/teja/hf TT_DIT_CACHE_DIR=/mnt/tt-data/teja/wan_cache WAN5B_SMOKE=1 PYTHONPATH=$ROOT
cd "$ROOT"; source python_env/bin/activate
echo "=== MANUAL SMOKE START $(date -u) ==="
echo "TT_DIT_CACHE_DIR=$TT_DIT_CACHE_DIR"; ls -la "$TT_DIT_CACHE_DIR" 2>/dev/null | head
python -m pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_ti2v_5b.py -k 4x8 -sv --timeout=2400
echo "=== MANUAL SMOKE EXIT=$? $(date -u) ==="
