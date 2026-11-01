"""ops.py <ops csv> <run log>: DEVICE KERNEL DURATION per op, named by the PASSED test ids of the run log."""
import csv, re, sys

path, log = sys.argv[1], sys.argv[2]
names = re.findall(r"::test_\w+\[([^\]]+)\]", open(log).read())
names = list(dict.fromkeys(names))
rows = list(csv.DictReader(open(path)))
vals = [int(r["DEVICE KERNEL DURATION [ns]"]) / 1000 for r in rows]
if len(names) != len(vals):
    names = [str(i) for i in range(len(vals))]
print("  " + "  ".join(f"{n.split('_')[0]}={v:.2f}" for n, v in zip(names, vals)))
