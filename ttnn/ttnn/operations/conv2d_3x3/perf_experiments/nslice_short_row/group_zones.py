# usage: group_zones.py <profile_log_device.csv> <run_index (1-based, runs with >= 8 cores)> [core_x,core_y]
# Per-group (compute_nsr_group) and weight-wait (compute_nsr_wwait) zone start / duration in us on one core (default:
# the core whose TRISC kernel ends last), per TRISC, relative to the run's earliest kernel start; plus medians over
# all cores of each group's duration per TRISC (the NSR_PROBE_GROUP_ZONE variants).
import collections
import csv
import statistics
import sys

path, idx = sys.argv[1], int(sys.argv[2])
want = tuple(int(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 else None
with open(path) as f:
    hdr = f.readline()
    freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = csv.reader(f)
    cols = [c.strip() for c in next(rows)]
    ix = {c: i for i, c in enumerate(cols)}
    runs = collections.defaultdict(list)
    for r in rows:
        if len(r) < len(cols):
            continue
        runs[int(r[ix["run host ID"]])].append(
            (
                (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                r[ix["RISC processor type"]].strip(),
                int(r[ix["time[cycles since reset]"]]),
                r[ix["zone name"]].strip(),
                r[ix["type"]].strip(),
            )
        )
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
ev = sorted(runs[big[idx - 1]], key=lambda e: e[2])
t0 = min(t for _, _, t, z, k in ev if z.endswith("-KERNEL") and k == "ZONE_START")
kend = collections.defaultdict(int)
spans = collections.defaultdict(list)  # (core, risc, zone) -> [(start, end)]
open_ = {}
for core, risc, t, z, kind in ev:
    if z.endswith("-KERNEL") and kind == "ZONE_END" and risc.startswith("TRISC"):
        kend[core] = max(kend[core], t)
    if z in ("compute_nsr_group", "compute_nsr_wwait", "compute_matmul_walk"):
        key = (core, risc, z)
        if kind == "ZONE_START":
            open_[key] = t
        elif key in open_:
            spans[key].append((open_.pop(key), t))
core = want or max(kend, key=kend.get)
us = lambda c: (c - t0) / freq
print(f"run {big[idx-1]} core {core}")
for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
    w = spans.get((core, risc, "compute_matmul_walk"), [])
    g = spans.get((core, risc, "compute_nsr_group"), [])
    ww = spans.get((core, risc, "compute_nsr_wwait"), [])
    print(f"  {risc} walk {[(round(us(a),2), round(us(b),2)) for a,b in w]}")
    print("    groups (start, dur):", " ".join(f"{us(a):.2f}+{(b-a)/freq:.3f}" for a, b in g))
    print("    wwait  (start, dur):", " ".join(f"{us(a):.2f}+{(b-a)/freq:.3f}" for a, b in ww))
# medians over cores, per group index, per TRISC
for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
    per = collections.defaultdict(list)
    for (c, r, z), sp in spans.items():
        if r == risc and z == "compute_nsr_group":
            for i, (a, b) in enumerate(sp):
                per[i].append((b - a) / freq)
    print(f"  median group dur {risc}:", " ".join(f"{statistics.median(per[i]):.3f}" for i in sorted(per)))
