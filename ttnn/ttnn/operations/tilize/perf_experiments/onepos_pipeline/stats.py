"""Median [min-max] and delta vs `head` per (case, variant) over `<variant>#<rep>` repeats.

usage: python stats.py <log>:<report_dir> ...
  <log>        stdout of a --profile run of test_tilize_perf2_onepos_pipeline.py (-s)
  <report_dir> that run's generated/profiler/reports/<ts>/ (the "SAFE_PYTEST: PROFILER CSV:" line);
               defaults to the dir named on that line in <log>
A variant's label is its token minus "#<rep>" (knob overrides stay part of the name).
"""
import csv, glob, os, re, statistics, sys
from collections import defaultdict

vals = defaultdict(list)
for arg in sys.argv[1:]:
    log, _, rep = arg.partition(":")
    text = open(log).read()
    if not rep:
        rep = os.path.dirname(re.findall(r"SAFE_PYTEST: PROFILER CSV: (\S+)", text)[-1])
    rows = list(csv.DictReader(open(glob.glob(os.path.join(rep, "ops_perf_results*.csv"))[0])))
    rows = [r for r in rows if "generic" in r["OP CODE"].lower() and r["DEVICE KERNEL DURATION [ns]"]]
    labels = []
    for m in re.finditer(r"^(?:ONEPOS (.*)\n)?P2 case=(\S+) variant=(\S+) nops=(\d+) done", text, re.M):
        # generic ops of this test: one "digest=" entry per program built (older logs: nops)
        n = m.group(1).count("digest=") if m.group(1) and "digest=" in m.group(1) else int(m.group(4))
        labels += [(m.group(2), re.sub(r"#\d+", "", m.group(3)))] * n
    labels = labels[-len(rows) :]
    assert len(labels) == len(rows), (len(labels), len(rows))
    for (c, v), r in zip(labels, rows):
        vals[(c, v)].append(int(float(r["DEVICE KERNEL DURATION [ns]"])))
cases = list(dict.fromkeys(c for c, _ in vals))
vs = list(dict.fromkeys(v for _, v in vals))
vs = (["head"] if "head" in vs else []) + [v for v in vs if v != "head"]  # head = the baseline column
w = max(26, max(len(v) for v in vs) + 2)
print(f"{'case':>18s} " + " ".join(f"{v:>{w}s}" for v in vs))
for c in cases:
    base = statistics.median(vals[(c, vs[0])]) if (c, vs[0]) in vals else None
    cells = []
    for v in vs:
        x = vals.get((c, v))
        if not x:
            cells.append(f"{'-':>{w}s}")
            continue
        m = statistics.median(x)
        d = f"{100 * (m / base - 1):+.1f}%" if base else ""
        cells.append(f"{int(m):>7d} [{min(x):>5d}-{max(x):>5d}] {d:>6s}".rjust(w))
    print(f"{c:>18s} " + " ".join(cells) + f"   n={len(vals[(c, vs[0])])}")
