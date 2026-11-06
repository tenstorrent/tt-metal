"""Label the --profile ops CSV's generic-op rows with the bench's call log. Usage: label_csv.py <csv> <log>
The precompile collect pass runs every test body once without dispatch, so when the log holds twice
the dispatched count, the second half is the real pass."""
import csv
import json
import sys

rows = [r for r in csv.DictReader(open(sys.argv[1])) if "generic" in r["OP CODE"].lower()]
labels = [json.loads(l) for l in open(sys.argv[2])]
if len(labels) >= 2 * len(rows) and len(labels) % 2 == 0:
    labels = labels[len(labels) // 2 :]
print(f"{len(rows)} generic rows, {len(labels)} labels")
for lab, r in zip(labels, rows):
    print(f"{lab['case']:34s} {lab['who']:5s} {float(r['DEVICE KERNEL DURATION [ns]']):10.0f}")
