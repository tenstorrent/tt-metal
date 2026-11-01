#!/bin/bash
# Usage: LN3P_MODE=<m> [LN_ABLATE=read,write] ./run.sh [--profile] [pytest node suffix]
# Runs test_3p.py (or one node) through run_safe_pytest.sh; prints precision, verdicts, errors.
cd /localdev/kgrujcic/tt-metal-codegen || exit 1
source python_env/bin/activate
prof=""
if [ "$1" == "--profile" ]; then prof="--profile"; shift; fi
node="ttnn/ttnn/operations/layer_norm_tile/perf_experiments/three_pass_restructure/test_3p.py$1"
timeout 1500 scripts/run_safe_pytest.sh $prof --run-all "$node" --import-mode=prepend -s 2>&1 \
  | grep -v "waiting for device" \
  | grep -E "LN3P|PASSED|FAILED|passed|failed|error:|TT_THROW|hang|triage|SAFE_PYTEST_ERROR" \
  | grep -v "^20.*BuildKernels" | cut -c1-400 | head -60
