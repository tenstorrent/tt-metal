#!/bin/bash
# T3K W-sweep: hybrid TQ + ring at W=128 (sweet spot from N150) on 8 chips.
# Plus Track A baseline at K=1 and K=14 for comparison.
set -uo pipefail

cd /localdev/mtairum/tt-metal
export PYTHONPATH=/localdev/mtairum/tt-metal
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
export TT_CACHE_PATH=/localdev/mtairum/hf/ttnn_cache/meta-llama/Llama-3.1-8B-Instruct
export HF_HOME=/localdev/proj_sw/user_dev/hf_data
export HF_TOKEN=${HF_TOKEN}
export TT_NUM_DEVICES=8

LOG_DIR=/tmp/t3k_sweep_logs
mkdir -p $LOG_DIR
RESULTS=/tmp/t3k_sweep_results.txt
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

# Track A baseline at K=1 (single-core SDPA per head)
run_one "t3k_track_A_K1"     "--tq-num-cores-per-head 1"
# Track A baseline at K=14 (Tier 2A perf path on T3K)
run_one "t3k_track_A_K14"    "--tq-num-cores-per-head 14"
# Hybrid W=128 (combine forces K=1 internally regardless of flag)
run_one "t3k_hybrid_W128"    "--tq-recent-window 128"

echo "==="
cat "$RESULTS"
