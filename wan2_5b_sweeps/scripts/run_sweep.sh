#!/usr/bin/env bash
cd /mnt/tt-data/teja/tt-metal
export TT_METAL_HOME=$PWD HF_HOME=/mnt/tt-data/teja/hf \
       TT_DIT_CACHE_DIR=/mnt/tt-data/teja/wan_cache \
       PYTHONPATH=$PWD:$(python3 -m site --user-site)
export WAN5B_FRAMES=81 WAN5B_STEPS=6 WAN5B_TRACED=1 WAN5B_REPEAT=2 WAN5B_FPS=24
source python_env/bin/activate
SUM=/home/ttuser/sweep_summary.txt
: > "$SUM"
for TC in 7 11 15 31; do
  echo "========== SWEEP vae_t_chunk=$TC ($(date)) =========="
  L=/home/ttuser/sweep_tc${TC}.log
  WAN5B_VAE_TCHUNK=$TC python -m pytest \
    models/tt_dit/tests/models/wan2_2/test_pipeline_wan_ti2v_5b.py \
    -k "generate and bh_4x8" -sv --timeout=3600 > "$L" 2>&1
  RC=$?
  WARM=$(grep "HOST_VS_DEVICE \[warm_traced\]" "$L" | tail -n1)
  OOM=$(grep -c "Out of Memory" "$L")
  echo "[TC=$TC rc=$RC oom=$OOM] $WARM" | tee -a "$SUM"
done
echo "=== SWEEP DONE $(date) ==="; echo "----- SUMMARY -----"; cat "$SUM"
