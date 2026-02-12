#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Run the VIT demo test locally on N300 hardware.
# This reproduces the vit-N300-func test from the single-card-demo-tests CI pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve TT_METAL_HOME: prefer env var, else infer from script location (parent of vit_n300 = repo root)
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

if [[ ! -d "${TT_METAL_HOME}" ]]; then
    echo "ERROR: TT_METAL_HOME directory not found: ${TT_METAL_HOME}" 1>&2
    echo "Set TT_METAL_HOME to your tt-metal repo root, or run this script from within the repo." 1>&2
    exit 1
fi

echo "[run_vit_n300] TT_METAL_HOME=${TT_METAL_HOME}"
echo "[run_vit_n300] Running VIT demo test (N300 single-card functional)..."
echo ""

# Match CI environment for single-card demo tests (see .github/workflows/single-card-demo-tests-impl.yaml)
export TT_METAL_HOME
export PYTHONPATH="${TT_METAL_HOME}"
export LD_LIBRARY_PATH="${TT_METAL_HOME}/build/lib:${LD_LIBRARY_PATH:-}"
export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

# Same test as run_vit_demo in tests/scripts/single_card/run_single_card_demo_tests.sh
TEST_FILE="models/demos/vision/classification/vit/wormhole/tests/test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py"

cd "${TT_METAL_HOME}"
pytest --disable-warnings -v -s "${TEST_FILE}"

echo ""
echo "[run_vit_n300] Test completed successfully."
