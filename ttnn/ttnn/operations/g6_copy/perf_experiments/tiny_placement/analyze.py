"""Map the ops CSV of a test_sweep log onto its TP_CFG lines; print median(launches 2..N) per config."""
import csv, re, statistics, sys

log = open(sys.argv[1], errors="replace").read()
csvp = re.search(r"PROFILER CSV: (\S+)", log).group(1)
rows = [r for r in csv.DictReader(open(csvp)) if r.get("DEVICE KERNEL DURATION [ns]")]
cfgs = re.findall(r"^TP_CFG (\d+) (\d+) (.*)$", log, re.M)
total = sum(int(c[1]) for c in cfgs)
print(f"# rows={len(rows)} expected={total} csv={csvp}")
assert len(rows) == total, "row count mismatch"
i = 0
for idx, n, label in cfgs:
    n = int(n)
    v = [float(r["DEVICE KERNEL DURATION [ns]"]) for r in rows[i : i + n]]
    i += n
    print(f"{statistics.median(v[1:]):8.0f}  {label}")
