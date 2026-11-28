"""parse.py <log> <case> <impl>: median DEVICE KERNEL DURATION of launches 2..N of the timed op (rows with a kernel
duration whose count matches the bench's launches: the last 12 such rows)."""
import sys, csv, statistics, re

log, c, impl = sys.argv[1:]
m = re.search(r"PROFILER CSV: (\S+)", open(log, errors="ignore").read())
rows = [r for r in csv.DictReader(open(m.group(1))) if r["DEVICE KERNEL DURATION [ns]"].strip()]
rows = rows[-12:]
d = [int(r["DEVICE KERNEL DURATION [ns]"]) for r in rows][1:]
print(
    f"RESULT {c} {impl} n={len(d)} median={statistics.median(d):.0f} min={min(d)} max={max(d)} nrows_total={len(rows)}"
)
