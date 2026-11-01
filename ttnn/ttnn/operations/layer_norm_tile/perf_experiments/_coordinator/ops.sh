#!/bin/bash
# Print DEVICE KERNEL DURATION + per-core min/max + per-RISC for the newest ops_perf CSV.
csv=${1:-$(ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)}
echo "$csv"
awk -F',' 'NR>1 {printf "kernel %6d | core min %6d max %6d | BR %6d NC %6d T0 %6d T1 %6d T2 %6d\n",$20,$22,$23,$26,$27,$28,$29,$30}' "$csv"
