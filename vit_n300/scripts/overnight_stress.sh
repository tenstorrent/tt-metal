#!/bin/bash
# Overnight stress test for ViT N300 ND hang reproduction.
# Runs the actual CI test in a loop with watcher + latency instrumentation.
#
# The matmul kernels are instrumented with wall-clock latency measurement
# at every semaphore wait site, pushing max-per-invocation to ring buffer.
# Ring buffer encoding: 0xSSNNNNNN (SS=site ID, NNNNNN=max cycles 24-bit)
#
# Phase 1: With debug delay on all cores (widens race windows, ~100s/run)
# Phase 2: Without debug delay, watcher only (~100s/run due to latency instrumentation)
#
# If a hang is detected, the script saves the watcher log and runs the
# latency parser on it, then stops.
#
# Usage: ./vit_n300/scripts/overnight_stress.sh
# Expected runtime: 8+ hours

set -uo pipefail

LOG_DIR="vit_n300/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_LOG="${LOG_DIR}/overnight_${TIMESTAMP}.log"

HANG_COUNT=0
PASS_COUNT=0
TOTAL_OPS=0

log() {
    echo "$1" | tee -a "$SUMMARY_LOG"
}

check_for_hang() {
    local run_log="$1"
    local run_num="$2"
    local phase="$3"

    if grep -q "TIMEOUT: device timeout" "$run_log"; then
        log "*** HANG DETECTED in phase $phase, run $run_num! ***"
        HANG_COUNT=$((HANG_COUNT + 1))

        # Save watcher log (contains ring buffer latency data)
        if [ -f "generated/watcher/watcher.log" ]; then
            local watcher_save="${LOG_DIR}/HANG_watcher_${phase}_run${run_num}_${TIMESTAMP}.log"
            cp "generated/watcher/watcher.log" "$watcher_save"
            log "Watcher log saved: $watcher_save"

            # Run latency parser on the hung watcher log
            log "--- Latency analysis of hung state ---"
            python vit_n300/scripts/parse_latency.py "$watcher_save" 2>&1 | tee -a "$SUMMARY_LOG"
        fi

        # Save the run log
        cp "$run_log" "${LOG_DIR}/HANG_run_${phase}_${run_num}_${TIMESTAMP}.log"
        return 1
    fi
    return 0
}

run_one_test() {
    local run_num="$1"
    local phase="$2"
    local run_log="${LOG_DIR}/overnight_run_${phase}_${run_num}_${TIMESTAMP}.log"

    python -u -m pytest \
        models/demos/vision/classification/vit/wormhole/tests/test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py::test_vit \
        -v -s -x --timeout=600 --tb=short -m "" \
        2>&1 > "$run_log"

    local exit_code=$?

    if check_for_hang "$run_log" "$run_num" "$phase"; then
        return 1
    fi

    PASS_COUNT=$((PASS_COUNT + 1))
    TOTAL_OPS=$((TOTAL_OPS + 66000))

    # Extract samples/sec for logging
    local sps=$(grep "Samples per second:" "$run_log" | grep -oP '[\d.]+' | tail -1)
    log "[$phase] Run $run_num: ${sps:-??} sps, total ops: $TOTAL_OPS"

    # Clean up individual run logs to save disk space (keep only last 5)
    if [ "$run_num" -gt 5 ]; then
        local old_log="${LOG_DIR}/overnight_run_${phase}_$((run_num - 5))_${TIMESTAMP}.log"
        rm -f "$old_log"
    fi

    return 0
}

# ============================================================================
log "=== Overnight ViT N300 Stress Test ==="
log "Started: $(date)"
log "Latency instrumentation: sites 0x10-0xA0 (see memory.md for encoding)"
log "MCAST_INPUT_BUFFERING_DEPTH: 2 (default, matching CI)"
log ""

# Phase 1: With debug delay (widens race windows)
PHASE1_RUNS=30
log "--- Phase 1: With debug delay (10000 cycles, write+atomic+read, all cores) ---"
log "Runs: $PHASE1_RUNS, ~120s each, total ~60 min"

export TT_METAL_WATCHER=1
export TT_METAL_WATCHER_DEBUG_DELAY=10000
export TT_METAL_WRITE_DEBUG_DELAY_CORES='(0,0)-(7,7)'
export TT_METAL_ATOMIC_DEBUG_DELAY_CORES='(0,0)-(7,7)'
export TT_METAL_READ_DEBUG_DELAY_CORES='(0,0)-(7,7)'
export TT_METAL_OPERATION_TIMEOUT_SECONDS=30

for i in $(seq 1 $PHASE1_RUNS); do
    if ! run_one_test "$i" "delay"; then
        log "Stopping Phase 1 due to hang."
        break
    fi
    sleep 2
done

if [ "$HANG_COUNT" -gt 0 ]; then
    log ""
    log "=== HANG FOUND in Phase 1! ==="
    log "Check ${LOG_DIR}/HANG_watcher_*.log for ring buffer latency data."
    log "Finished: $(date)"
    exit 1
fi

# Phase 2: No delay, watcher only (for ring buffer capture if hang occurs)
PHASE2_RUNS=200
log ""
log "--- Phase 2: No delay, watcher only (~100s/run with latency overhead) ---"
log "Runs: $PHASE2_RUNS"

unset TT_METAL_WATCHER_DEBUG_DELAY
unset TT_METAL_WRITE_DEBUG_DELAY_CORES
unset TT_METAL_ATOMIC_DEBUG_DELAY_CORES
unset TT_METAL_READ_DEBUG_DELAY_CORES
export TT_METAL_WATCHER=1
export TT_METAL_OPERATION_TIMEOUT_SECONDS=10

for i in $(seq 1 $PHASE2_RUNS); do
    if ! run_one_test "$i" "fast"; then
        log "Stopping Phase 2 due to hang."
        break
    fi
    sleep 2
done

if [ "$HANG_COUNT" -gt 0 ]; then
    log ""
    log "=== HANG FOUND in Phase 2! ==="
    log "Check ${LOG_DIR}/HANG_watcher_*.log for ring buffer latency data."
else
    log ""
    log "=== No hang detected ==="
fi

# Run final latency analysis on last passing watcher log
log ""
log "--- Final latency profile (last passing run) ---"
if [ -f "generated/watcher/watcher.log" ]; then
    python vit_n300/scripts/parse_latency.py "generated/watcher/watcher.log" 2>&1 | tee -a "$SUMMARY_LOG"
fi

log ""
log "=== Final Summary ==="
log "Total passes: $PASS_COUNT"
log "Total hangs: $HANG_COUNT"
log "Total matmul ops: $TOTAL_OPS"
log "Finished: $(date)"
