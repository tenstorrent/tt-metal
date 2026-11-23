#!/bin/bash
# usage: env... zp.sh label
cd /home/malimpic/localdev/2026_11_23/1317_malimpic_distributed_rms_norm_all_gather_gen/clones/all_gather_rms_norm_run1/tt-metal
source python_env/bin/activate
T=tests/ttnn/unit_tests/operations/all_gather_rms_norm/perf_experiments/own_entry_port_l1_staging
AGRN_PROFILE_ZONES=1 timeout 900 scripts/run_safe_pytest.sh --profile --no-precompile $T/probe_stage.py -k "${PROBE_K:-focus}" > /tmp/zp_$1.log 2>&1
grep "SAFE_PYTEST_RESULT\|passed\|failed" /tmp/zp_$1.log | tail -2
python tests/ttnn/unit_tests/operations/all_gather_rms_norm/zone_timeline.py > /tmp/zt_$1.txt 2>&1
grep "===\|wr_wait_out\|wr_stage\|rd_stage\|wr_own \|pr_own_first\|ps_own_pkt0\|ps_relay_start\|pr_wait_all\|ps_bulk\|BRISC-KERNEL" /tmp/zt_$1.txt | head -${ZN:-24}
