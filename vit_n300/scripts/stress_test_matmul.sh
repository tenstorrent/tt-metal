#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  Matmul Deadlock Stress Test
#
#  Runs the block-sharded matmul spam test in a loop with device resets
#  between runs.  Targets the exact Tensix deadlock found in CI triage:
#    bmm_large_block_zm_fused_bias_activation on 8x8 grid with multicast
#
#  Test variants:
#    1. traced    - trace replay of 4 matmuls x 10000 iters (fast)
#    2. direct    - direct execution of 4 matmuls x 10000 iters
#    3. 2cq       - trace replay + concurrent CQ1 copies (matches CI failure)
#    4. wide      - wider matmuls for more back-pressure (direct only)
#
#  Usage:
#    ./vit_n300/scripts/stress_test_matmul.sh               # run all variants
#    ./vit_n300/scripts/stress_test_matmul.sh --2cq-only     # only 2CQ variant
#    ./vit_n300/scripts/stress_test_matmul.sh --traced-only   # only traced variant
#    ./vit_n300/scripts/stress_test_matmul.sh --wide-only     # only wide variant
#    ./vit_n300/scripts/stress_test_matmul.sh --dprint        # enable DPRINT CB monitor
#    ./vit_n300/scripts/stress_test_matmul.sh --2cq-only --dprint  # combine flags
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

if [[ ! -d "${TT_METAL_HOME}" ]]; then
    echo "ERROR: TT_METAL_HOME directory not found: ${TT_METAL_HOME}" 1>&2
    exit 1
fi

LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/stress_matmul_$(date +%Y%m%d_%H%M%S).log"

DURATION_SEC=3600     # 1 hour default
TEST_FILE="vit_n300/tests/test_matmul_deadlock_stress.py"

export TT_METAL_HOME
export PYTHONPATH="${TT_METAL_HOME}"
export LD_LIBRARY_PATH="${TT_METAL_HOME}/build/lib:${LD_LIBRARY_PATH:-}"
export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

# Parse arguments
TEST_FILTER=""
DPRINT_ENABLED=false
for arg in "$@"; do
    case "${arg}" in
        --2cq-only)    TEST_FILTER="-k test_matmul_deadlock_stress_2cq" ;;
        --traced-only) TEST_FILTER="-k traced" ;;
        --direct-only) TEST_FILTER="-k direct" ;;
        --wide-only)   TEST_FILTER="-k wide" ;;
        --dprint)      DPRINT_ENABLED=true ;;
    esac
done

# Enable DPRINT CB monitoring if requested
if [[ "${DPRINT_ENABLED}" == "true" ]]; then
    DPRINT_FILE="${LOG_DIR}/dprint_$(date +%Y%m%d_%H%M%S).log"
    # Monitor 1 receiver core with dataflow RISCs only.
    # (7,7) is a receiver for both in0 and in1 multicast — the contention hotspot.
    # Compute kernel DPRINT removed (static counters reset per trace replay, useless).
    # Event-driven: only prints when contention detected (sem=INVALID).
    export TT_METAL_DPRINT_CORES="(7,7)"
    export TT_METAL_DPRINT_RISCVS="BR+NC"
    export TT_METAL_DPRINT_FILE="${DPRINT_FILE}"
fi

# ── logging helper ──
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }

log "=== Matmul Deadlock Stress Test ==="
log "TT_METAL_HOME : ${TT_METAL_HOME}"
log "Test file     : ${TEST_FILE}"
log "Filter        : ${TEST_FILTER:-<all>}"
log "Log file      : ${LOG_FILE}"
log "Duration cap  : ${DURATION_SEC}s"
if [[ "${DPRINT_ENABLED}" == "true" ]]; then
    log "DPRINT        : ENABLED"
    log "DPRINT cores  : ${TT_METAL_DPRINT_CORES}"
    log "DPRINT RISCs  : ${TT_METAL_DPRINT_RISCVS}"
    log "DPRINT file   : ${DPRINT_FILE}"
else
    log "DPRINT        : disabled (use --dprint to enable)"
fi
log ""

PASS=0
FAIL=0
RUN=0
START_EPOCH=$(date +%s)

while true; do
    NOW_EPOCH=$(date +%s)
    ELAPSED=$(( NOW_EPOCH - START_EPOCH ))
    if (( ELAPSED >= DURATION_SEC )); then
        log "Time limit reached (${DURATION_SEC}s).  Stopping."
        break
    fi

    RUN=$(( RUN + 1 ))
    log "──── Run ${RUN}  (elapsed ${ELAPSED}s / ${DURATION_SEC}s) ────"

    set +e
    (
        cd "${TT_METAL_HOME}"
        python -m pytest "${TEST_FILE}" ${TEST_FILTER} -v -s -x --timeout=600 2>&1
    ) | tee -a "${LOG_FILE}"
    RC=${PIPESTATUS[0]}
    set -e

    if [[ ${RC} -eq 0 ]]; then
        PASS=$(( PASS + 1 ))
        log "Run ${RUN}: PASS  (total: ${PASS} pass, ${FAIL} fail)"
    else
        FAIL=$(( FAIL + 1 ))
        log "Run ${RUN}: FAIL (exit ${RC})  (total: ${PASS} pass, ${FAIL} fail)"
        log "*** FAILURE DETECTED — check logs above for triage data ***"
        if [[ "${DPRINT_ENABLED}" == "true" ]]; then
            log "DPRINT output saved to: ${DPRINT_FILE}"
        fi
    fi

    log "Resetting device with tt-smi -r ..."
    tt-smi -r 0 2>&1 | tee -a "${LOG_FILE}" || true
    sleep 5
    break
done

log ""
log "=== Summary ==="
log "Total runs: ${RUN}   Pass: ${PASS}   Fail: ${FAIL}"
log "Log saved to: ${LOG_FILE}"
if [[ "${DPRINT_ENABLED}" == "true" ]]; then
    log "DPRINT output: ${DPRINT_FILE}"
fi

if (( FAIL > 0 )); then
    log "FAILURES DETECTED — review log for details."
    exit 1
fi
