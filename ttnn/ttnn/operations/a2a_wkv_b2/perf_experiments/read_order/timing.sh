#!/bin/bash
# read_order: usage timing.sh [env assignments...] — trace-replay us/call of the read_order variant.
D=ttnn/ttnn/operations/a2a_wkv_b2/perf_experiments/read_order
cd /data/dstoiljkovic/tt-metal
/data/dstoiljkovic/bin/on-node "env $* scripts/run_safe_pytest.sh $D/test_ro_trace_timing.py -o addopts= --import-mode=prepend -s" 2>&1 | grep -E "TRACE_TIMING|passed|failed|rror" | tail -3
