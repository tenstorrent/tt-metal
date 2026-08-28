#!/bin/bash
# The exit-gate line "existing 1D model contract and demo-contract host tests
# green, expectations unchanged", as its own selection.
#
# Explicit file list, not a directory: the 1D model directories also hold
# `test_p150x4_smoke.py`, `test_t3k_batched_prefill_correctness.py` and
# `test_model_profile.py`, which open real devices. This job holds the Galaxy
# exclusively, so a gate that quietly takes the mesh is not a host gate.
set -u
LOG="$1"; shift
export HF_HOME=/localdev/ctr-apbernal/hf_data
timeout --signal=TERM --kill-after=60 1800 python -m pytest -q -rA --color=no -p no:cacheprovider \
  $(ls models/common/tests/models/*/test_demo_contract.py models/common/tests/models/*/test_hf_adaptor.py 2>/dev/null | grep -v galaxy) \
  "$@" > "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
tail -3 "$LOG"
