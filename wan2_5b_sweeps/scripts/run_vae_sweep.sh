#!/usr/bin/env bash
# VAE-decode chunk-size sweep at 81 frames (5B), 480p + 720p.
# latent T at 81f = (81-1)/4 + 1 = 21, so t_chunk >= 21 == full-T (single chunk).
# Denoise kept tiny (6 steps) -- we only care about vae_decode time here.
cd /mnt/tt-data/teja/tt-metal
export TT_METAL_HOME=$PWD HF_HOME=/mnt/tt-data/teja/hf \
       TT_DIT_CACHE_DIR=/mnt/tt-data/teja/wan_cache \
       PYTHONPATH=$PWD:$(python3 -m site --user-site)
export WAN5B_FRAMES=81 WAN5B_STEPS=6 WAN5B_TRACED=1 WAN5B_REPEAT=2 WAN5B_FPS=24 WAN5B_CLIP=0
source python_env/bin/activate
SUM=/home/ttuser/vae_sweep_summary.txt
: > "$SUM"
echo "res       tchunk  rc  oom  vae_decode / e2e (warm)" | tee -a "$SUM"
run () {
  local W=$1 H=$2 TAG=$3
  for TC in 7 11 21 31; do
    local L=/home/ttuser/vaesweep_${TAG}_tc${TC}.log
    echo "========== ${TAG} ${W}x${H} vae_t_chunk=$TC ($(date)) =========="
    WAN5B_WIDTH=$W WAN5B_HEIGHT=$H WAN5B_VAE_TCHUNK=$TC \
      python -m pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_ti2v_5b.py \
        -k "generate and bh_4x8" -sv --timeout=3600 > "$L" 2>&1
    local RC=$?
    local OOM=$(grep -c "Out of Memory" "$L")
    local WARM=$(grep "HOST_VS_DEVICE \[warm_traced\]" "$L" | tail -n1 | sed -E "s/.*(e2e=.*)/\1/")
    printf "%-9s %-6s %-3s %-4s %s\n" "$TAG" "$TC" "$RC" "$OOM" "$WARM" | tee -a "$SUM"
  done
}
run 832  480  480p
run 1280 704  720p
echo "=== VAE SWEEP DONE $(date) ==="; echo "----- SUMMARY -----"; cat "$SUM"
