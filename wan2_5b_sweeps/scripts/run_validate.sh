#!/usr/bin/env bash
set -o pipefail
cd /mnt/tt-data/teja/tt-metal
export TT_METAL_HOME=$PWD HF_HOME=/mnt/tt-data/teja/hf \
       TT_DIT_CACHE_DIR=/mnt/tt-data/teja/wan_cache \
       PYTHONPATH=$PWD:$(python3 -m site --user-site)
source python_env/bin/activate

echo "===== [1/2] VAE full-T vs chunked PCC ($(date)) ====="
WAN5B_PCC_FRAMES=${WAN5B_PCC_FRAMES:-57} WAN5B_PCC_TCHUNK=${WAN5B_PCC_TCHUNK:-7} \
WAN5B_PCC_HEIGHT=${WAN5B_PCC_HEIGHT:-256} WAN5B_PCC_WIDTH=${WAN5B_PCC_WIDTH:-512} \
  python -m pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_ti2v_5b.py \
  -k "chunk_pcc and bh_4x8" -sv --timeout=1800 2>&1 | tee /home/ttuser/validate_pcc.log
echo "PCC rc=${PIPESTATUS[0]}"

echo "===== [2/2] CLIP-gated T2V generate 121f/40steps ($(date)) ====="
WAN5B_FRAMES=${WAN5B_FRAMES:-121} WAN5B_STEPS=${WAN5B_STEPS:-40} \
WAN5B_TRACED=${WAN5B_TRACED:-1} WAN5B_REPEAT=${WAN5B_REPEAT:-1} WAN5B_CLIP=1 \
WAN5B_OUT=/home/ttuser/wan5b_validate_121f.mp4 \
  python -m pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_ti2v_5b.py \
  -k "generate and bh_4x8" -sv --timeout=3600 2>&1 | tee /home/ttuser/validate_gen.log
echo "GEN rc=${PIPESTATUS[0]}"

echo "=== VALIDATE DONE $(date) ==="
echo "----- CLIP line -----"; grep -E "CLIP scores" /home/ttuser/validate_gen.log | tail -n1
echo "----- PCC line -----";  grep -E "VAE_CHUNK_PCC" /home/ttuser/validate_pcc.log | tail -n1
