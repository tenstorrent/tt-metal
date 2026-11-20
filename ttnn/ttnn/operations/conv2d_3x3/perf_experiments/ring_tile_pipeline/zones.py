# usage: zones.py <profile_log_device.csv[.gz]> <run_index (1-based, runs with >= 8 cores)> [core_x,core_y]
# Per-step / per-group probe zones (RTP_PROBE_ZONE) on one core (default: the core whose TRISC kernel ends last),
# us relative to the run's earliest kernel start: unpack weight wait, math DEST acquire + matmul, pack arrival wait +
# math wait + pack, reader token wait + send. Then medians over all cores of each zone's duration.
import collections
import csv
import gzip
import statistics
import sys

path, idx = sys.argv[1], int(sys.argv[2])
want = tuple(int(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 else None
opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt") as f:
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
spans = collections.defaultdict(list)
open_ = {}
for core, risc, t, z, kind in ev:
    if z.endswith("-KERNEL") and kind == "ZONE_END" and risc.startswith("TRISC"):
        kend[core] = max(kend[core], t)
    key = (core, risc, z)
    if kind == "ZONE_START":
        open_[key] = t
    elif key in open_:
        spans[key].append((open_.pop(key), t))
core = want or max(kend, key=kend.get)
us = lambda c: (c - t0) / freq
print(f"run {big[idx-1]} core {core} TRISC end {us(kend[core]):.2f} us")
show = [
    ("TRISC_0", "rtp_wwait"),
    ("TRISC_1", "rtp_acq"),
    ("TRISC_1", "rtp_math"),
    ("TRISC_2", "rtp_arr"),
    ("TRISC_2", "rtp_mwait"),
    ("TRISC_2", "rtp_pack"),
    ("NCRISC", "reader_ring_wait_step"),
    ("NCRISC", "reader_ring_send"),
    ("TRISC_0", "compute_matmul_walk"),
    ("TRISC_2", "compute_combine"),
]
for risc, z in show:
    sp = spans.get((core, risc, z), [])
    if sp:
        print(f"  {risc:7} {z:22} " + " ".join(f"{us(a):.2f}+{(b-a)/freq:.3f}" for a, b in sp))
print("medians over cores (per occurrence index: dur us):")
for risc, z in show:
    per = collections.defaultdict(list)
    for (c, r, zz), sp in spans.items():
        if r == risc and zz == z:
            for i, (a, b) in enumerate(sp):
                per[i].append((b - a) / freq)
    if per:
        print(f"  {risc:7} {z:22} " + " ".join(f"{statistics.median(v):.3f}" for i, v in sorted(per.items())))
