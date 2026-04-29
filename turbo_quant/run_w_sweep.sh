#!/bin/bash
# W-sweep: hybrid TQ + ring at W ∈ {32, 64, 128, 256}, full 512-position eval.
# Plus baseline Track A (W=0) and BFP8 baseline for comparison.
set -uo pipefail

cd /localdev/mtairum/tt-metal
export PYTHONPATH=/localdev/mtairum/tt-metal
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
export TT_CACHE_PATH=/localdev/mtairum/hf/ttnn_cache/meta-llama/Llama-3.1-8B-Instruct
export HF_HOME=/localdev/proj_sw/user_dev/hf_data
export HF_TOKEN=${HF_TOKEN}

LOG_DIR=/tmp/w_sweep_logs
mkdir -p $LOG_DIR
RESULTS=/tmp/w_sweep_results.txt
> "$RESULTS"

run_one() {
  local label="$1"
  local extra_args="$2"
  local log="$LOG_DIR/${label}.log"
  echo "=== Running: $label ($extra_args) ==="
  python -u turbo_quant/eval_token_accuracy.py --tq-full-dequant --max-seq-len 1024 $extra_args > "$log" 2>&1
  local ec=$?
  if [[ $ec -ne 0 ]]; then
    echo "  FAIL exit=$ec — see $log"
    echo "$label: FAIL (exit=$ec)" >> "$RESULTS"
    return 1
  fi
  local top1=$(grep "Top-1 accuracy" "$log" | tail -1 | awk -F': ' '{print $2}')
  local top5=$(grep "Top-5 accuracy" "$log" | tail -1 | awk -F': ' '{print $2}')
  local lat=$(grep "Avg latency" "$log" | tail -1 | awk -F': ' '{print $2}')
  echo "  top-1: $top1   top-5: $top5   latency: $lat"
  echo "$label: top-1=$top1  top-5=$top5  $lat" >> "$RESULTS"
}

run_one "track_A_W0" ""
run_one "hybrid_W32"  "--tq-recent-window 32"
run_one "hybrid_W64"  "--tq-recent-window 64"
run_one "hybrid_W128" "--tq-recent-window 128"
run_one "hybrid_W256" "--tq-recent-window 256"

echo "==="
cat "$RESULTS"
