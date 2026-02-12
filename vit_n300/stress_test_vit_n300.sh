#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Stress test the VIT N300 demo to reproduce the non-deterministic fetch-queue timeout.
# Runs for ~30 minutes or until first failure. On failure, stops immediately so you get
# the full error and triage output.
#
# Usage: ./stress_test_vit_n300.sh
#
# Output is written to vit_n300/logs/stress_YYYYMMDD_HHMMSS.log and to the terminal.
# Monitor with: tail -f vit_n300/logs/stress_*.log (or the specific file).
#
# If no failure in 30 min: consider modifying the test to increase operations
# (e.g. more iterations) to amplify the probability of hitting the bug.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ ! -d "${TT_METAL_HOME}" ]]; then
    echo "ERROR: TT_METAL_HOME directory not found: ${TT_METAL_HOME}" 1>&2
    exit 1
fi

# Log directory and file (one per stress-test session)
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/stress_$(date +%Y%m%d_%H%M%S).log"

# Stress test parameters
DURATION_SEC=1800   # 30 minutes
TEST_FILE="models/demos/vision/classification/vit/wormhole/tests/test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py"

# Environment (match CI + run_vit_n300.sh)
export TT_METAL_HOME
export PYTHONPATH="${TT_METAL_HOME}"
export LD_LIBRARY_PATH="${TT_METAL_HOME}/build/lib:${LD_LIBRARY_PATH:-}"
export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

# Keep default 5s timeout so hangs fail fast; triage runs on timeout
# Uncomment to allow more time for triage if needed:
# export TT_METAL_OPERATION_TIMEOUT_SECONDS=30

cd "${TT_METAL_HOME}"

# All output goes to log file and terminal
exec > >(tee -a "${LOG_FILE}") 2>&1

START=$(date +%s)
RUN=0

echo "========================================================================"
echo "[stress] VIT N300 stress test — target ${DURATION_SEC}s (~30 min)"
echo "[stress] Stop on first failure (-x). Test: ${TEST_FILE}"
echo "[stress] Log: ${LOG_FILE}"
echo "========================================================================"

while true; do
    ELAPSED=$(($(date +%s) - START))
    if [[ $ELAPSED -ge $DURATION_SEC ]]; then
        echo ""
        echo "[stress] Completed ${RUN} runs in ${ELAPSED}s with no failure."
        echo "[stress] Consider increasing iterations in the test to stress the pipeline further."
        exit 0
    fi

    RUN=$((RUN + 1))
    echo "[stress] Run ${RUN} (elapsed ${ELAPSED}s)..."

    if ! pytest --disable-warnings -v -s -x "${TEST_FILE}"; then
        echo ""
        echo "[stress] FAILED on run ${RUN} after ${ELAPSED}s."
        exit 1
    fi

    # Reset device between runs to replicate CI (fresh environment each test)
    tt-smi -r
done
