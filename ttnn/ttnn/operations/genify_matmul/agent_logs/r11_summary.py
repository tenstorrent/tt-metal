"""Summarize genify_matmul device kernel ns from an ops_perf_results CSV, in test order.
usage: r11_summary.py <csv> [name1 name2 ...]  (names label the GenericOp rows in order)"""
import csv, sys

rows = list(csv.DictReader(open(sys.argv[1])))
names = sys.argv[2:]
ops = [r for r in rows if r["OP CODE"].startswith("GenericOp")]
for i, r in enumerate(ops):
    label = names[i] if i < len(names) else str(i)
    print(f"{label:14s} cores={r['CORE COUNT']:>4s} kernel_us={int(r['DEVICE KERNEL DURATION [ns]'])/1000:9.1f}")
