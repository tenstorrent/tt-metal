#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Stress test the COPY-STRESS variant: 10 copies per iteration, 2000 iterations.
# ~21,000 stall points per run (vs ~1,100 in original) to amplify the ND failure.
#
# Usage: ./vit_n300/stress_test_copy_stress.sh
#
# Output: vit_n300/logs/stress_copy_YYYYMMDD_HHMMSS.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ ! -d "${TT_METAL_HOME}" ]]; then
    echo "ERROR: TT_METAL_HOME directory not found: ${TT_METAL_HOME}" 1>&2
    exit 1
fi

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/stress_copy_$(date +%Y%m%d_%H%M%S).log"

DURATION_SEC=1800
TEST_FILE="vit_n300/test_vit_2cq_copy_stress.py"

export TT_METAL_HOME
export PYTHONPATH="${TT_METAL_HOME}"
export LD_LIBRARY_PATH="${TT_METAL_HOME}/build/lib:${LD_LIBRARY_PATH:-}"
export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

cd "${TT_METAL_HOME}"
exec > >(tee -a "${LOG_FILE}") 2>&1

START=$(date +%s)
RUN=0

echo "========================================================================"
echo "[stress] VIT 2CQ COPY-STRESS — 10 copies/iter × 2000 iters ≈21K stalls/run"
echo "[stress] Target ${DURATION_SEC}s (~30 min). Log: ${LOG_FILE}"
echo "========================================================================"

while true; do
    ELAPSED=$(($(date +%s) - START))
    if [[ $ELAPSED -ge $DURATION_SEC ]]; then
        echo ""
        echo "[stress] Completed ${RUN} runs with no failure."
        exit 0
    fi

    RUN=$((RUN + 1))
    echo "[stress] Run ${RUN} (elapsed ${ELAPSED}s)..."

    if ! pytest --disable-warnings -v -s -x "${TEST_FILE}"; then
        echo ""
        echo "[stress] FAILED on run ${RUN} after ${ELAPSED}s."
        exit 1
    fi

    tt-smi -r
done
