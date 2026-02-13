#!/bin/bash
set -o pipefail
MAX_RUNS=${1:-200}
EXTRA_ENV="${2:-}"
LOGDIR="/tt-metal/vit_n300/logs/repro_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
PASS=0; FAIL=0; HANG=0
START_TIME=$(date +%s)
echo "=== ViT N300 Hang Reproduction Loop ==="
echo "Max runs: $MAX_RUNS | Extra env: $EXTRA_ENV | Log dir: $LOGDIR"
echo "Started: $(date)"
for i in $(seq 1 $MAX_RUNS); do
    RUN_START=$(date +%s)
    ELAPSED_TOTAL=$(( RUN_START - START_TIME ))
    echo "--- Run $i/$MAX_RUNS (pass=$PASS fail=$FAIL hang=$HANG, ${ELAPSED_TOTAL}s elapsed) ---"
    env TT_METAL_OPERATION_TIMEOUT_SECONDS=5 $EXTRA_ENV \
        timeout 120 pytest \
        models/demos/vision/classification/vit/wormhole/tests/test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py::test_vit \
        --no-header -rN -x \
        > "$LOGDIR/run_${i}.log" 2>&1
    EXIT_CODE=$?
    RUN_END=$(date +%s)
    RUN_DURATION=$(( RUN_END - RUN_START ))
    if [ $EXIT_CODE -eq 0 ]; then
        PASS=$((PASS + 1)); echo "  PASS (${RUN_DURATION}s)"
    elif [ $EXIT_CODE -eq 124 ]; then
        HANG=$((HANG + 1))
        echo "  *** HANG (outer timeout) *** (${RUN_DURATION}s) Log: $LOGDIR/run_${i}.log"
        cp "$LOGDIR/run_${i}.log" "$LOGDIR/HANG_run_${i}.log"
        tt-smi -r 0 > /dev/null 2>&1; sleep 3
    else
        if grep -q "device timeout\|Timeout.*fetch\|fetch queue wait" "$LOGDIR/run_${i}.log" 2>/dev/null; then
            HANG=$((HANG + 1))
            echo "  *** DEVICE HANG *** (exit=$EXIT_CODE, ${RUN_DURATION}s) Log: $LOGDIR/run_${i}.log"
            cp "$LOGDIR/run_${i}.log" "$LOGDIR/HANG_run_${i}.log"
            tt-smi -r 0 > /dev/null 2>&1; sleep 3
        else
            FAIL=$((FAIL + 1)); echo "  FAIL (exit=$EXIT_CODE, ${RUN_DURATION}s)"
            grep -q "Samples per second" "$LOGDIR/run_${i}.log" 2>/dev/null && echo "    (perf assertion)"
        fi
    fi
done
END_TIME=$(date +%s); TOTAL_TIME=$(( END_TIME - START_TIME ))
echo ""; echo "=== RESULTS ==="
echo "Runs: $MAX_RUNS | Pass: $PASS | Fail: $FAIL | HANG: $HANG | Time: ${TOTAL_TIME}s"
HANG_RATE=$(echo "scale=1; $HANG * 100 / ($PASS + $FAIL + $HANG)" | bc 2>/dev/null || echo "N/A")
echo "Hang rate: ${HANG_RATE}%"
echo "Runs=$MAX_RUNS Pass=$PASS Fail=$FAIL HANG=$HANG Time=${TOTAL_TIME}s Rate=${HANG_RATE}%" > "$LOGDIR/summary.txt"
