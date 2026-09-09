#!/usr/bin/env bash
# Run Wan2.2 on the quad Blackhole Galaxy.
# Run on host 13 (10.81.14.13).
#
#   ./run_wan.sh        one generation
#   ./run_wan.sh 8      eight sequential generations

set -euo pipefail

# ---- edit these (or export them before running; the shell value wins) -----
WAN_MODE=${WAN_MODE:-t2v}                                        # t2v or i2v
WAN_CKPT=${WAN_CKPT:-/mnt/tt-data/wan2.2/Wan2.2-T2V-A14B-Diffusers}
WAN_DIT_CACHE_DIR=${WAN_DIT_CACHE_DIR:-$HOME/dit_cache_wan_t2v}
WAN_STEPS=${WAN_STEPS:-40}
WAN_PROMPT=${WAN_PROMPT:-'A red fox trotting through a snowy forest at sunrise, cinematic style'}
WAN_IMAGE=${WAN_IMAGE:-/mnt/tt-data/wan2.2/prompt_image.png}     # i2v only
# --------------------------------------------------------------------------

COUNT=${1:-1}
OUT_DIR=$HOME/wan_runs/$(date +%Y%m%d-%H%M%S)
TT_ROOT=/mnt/tt-data/tt-metal
PRODUCED=$TT_ROOT/wan_output_video_${WAN_MODE}_traced.mp4

export WAN_MODE WAN_CKPT WAN_DIT_CACHE_DIR WAN_STEPS WAN_PROMPT WAN_IMAGE
export HF_HOME=${HF_HOME:-$HOME/hf_home}

mkdir -p "$OUT_DIR"

for i in $(seq 1 "$COUNT"); do
  # Clear any previous output so a failed run cannot copy a stale video.
  rm -f "$PRODUCED"

  # Run the generation. Capture pytest's real exit status (not tee's).
  set +e
  tt-run \
    --rank-binding tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml \
    --tcp-interface ens5f0np0 \
    --mpi-args "--host 10.81.14.13,10.81.14.14,10.81.14.15,10.81.14.16 --bind-to none --tag-output --mca oob_tcp_if_include 10.81.0.0/20" \
    bash -c "cd $TT_ROOT && \
    source $TT_ROOT/python_env/bin/activate && \
    export TT_METAL_HOME=$TT_ROOT && \
    export PYTHONPATH=$TT_ROOT && \
    export TT_DIT_CACHE_DIR=$WAN_DIT_CACHE_DIR && \
    export HF_HOME=$HF_HOME && \
    export WAN_CKPT=$WAN_CKPT && \
    export WAN_PROMPT=\"$WAN_PROMPT\" && \
    export WAN_IMAGE=$WAN_IMAGE && \
    export WAN_STEPS=$WAN_STEPS && \
    export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 && \
    export TT_METAL_OPERATION_TIMEOUT_SECONDS=120 && \
    pytest -v --timeout 3600 \
    models/tt_dit/tests/models/wan2_2/test_performance_wan.py \
    -k \"bh_4x32sp1tp0 and resolution_720p and $WAN_MODE\"" 2>&1 | tee "$OUT_DIR/gen_$i.log"
  rc=${PIPESTATUS[0]}
  set -e

  if [ "$rc" -ne 0 ]; then
    echo "WARNING: generation $i FAILED (pytest exit $rc). See $OUT_DIR/gen_$i.log" | tee -a "$OUT_DIR/gen_$i.log"
  fi

  # Only copy a genuinely fresh video (the file was removed above).
  if [ -f "$PRODUCED" ]; then
    cp "$PRODUCED" "$OUT_DIR/gen_$i.mp4"
    echo "Saved gen_$i.mp4"
  else
    echo "WARNING: no video produced for generation $i. See $OUT_DIR/gen_$i.log" | tee -a "$OUT_DIR/gen_$i.log"
  fi
done

echo "Results in $OUT_DIR"
