# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Print one DEVICE KERNEL DURATION row per rms_norm dispatch from a Tracy ops CSV, in dispatch order,
labelled with the guard-set cell ids (same order as test_rms_norm_perf_guard.CELLS). Host-only.

    python3 ttnn/ttnn/operations/rms_norm/perf_experiments/guard_report.py <ops_perf_results.csv> [labels.txt]
"""

import csv
import sys

CELLS = [
    "R2_decode_7168_focus",
    "R3_decode_5120_sharded_8x4",
    "R3_decode_7168_sharded_7x4",
    "R3_2048_sharded_8_fp32dest",
    "R1_prefill_1024",
    "R1_prefill_7168",
    "R1_4d_no_gamma_fp32dest",
    "R2_residency_64x12288_fp32dest",
    "R2_rm_x_rm_gamma",
    "R1_rm_x_rm_gamma_fp32_16bit",
    "R1_fp32_x_fp32_gamma",
    "R2_bf8b_x_bf8b_gamma",
]


def main():
    path = sys.argv[1]
    with open(path) as f:
        rows = list(csv.DictReader(f))
    ops = [r for r in rows if r["OP CODE"].strip() == "GenericOpDeviceOperation"]
    ops.sort(key=lambda r: int(r["GLOBAL CALL COUNT"]))
    labels = CELLS if len(ops) == len(CELLS) else [f"dispatch_{i}" for i in range(len(ops))]
    if len(ops) != len(CELLS):
        print(f"WARNING: {len(ops)} rms_norm dispatches, expected {len(CELLS)}")
    print(f"{'cell':36s} {'cores':>5s} {'kernel_ns':>10s} {'brisc':>8s} {'ncrisc':>8s} {'trisc1':>8s}")
    for lab, r in zip(labels, ops):
        print(
            f"{lab:36s} {r['CORE COUNT']:>5s} {r['DEVICE KERNEL DURATION [ns]']:>10s} "
            f"{r['DEVICE BRISC KERNEL DURATION [ns]']:>8s} {r['DEVICE NCRISC KERNEL DURATION [ns]']:>8s} "
            f"{r['DEVICE TRISC1 KERNEL DURATION [ns]']:>8s}"
        )


if __name__ == "__main__":
    main()
