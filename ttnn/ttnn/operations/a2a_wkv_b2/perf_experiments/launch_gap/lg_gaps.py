"""Per chip: the variant's program instances, each one's span and the idle gap before it.
Program i on a (core, risc) = its i-th *-KERNEL zone; the variant's keys are those with the most zones (the
pad trace runs on a subset or the same set), and its programs are their last K*R zones.
Usage: python3 lg_gaps.py <profile_log_device.csv> <K> <R>   (gaps at replay edges marked '|')"""
import csv, sys
from collections import defaultdict

F = 1350.0
path, K, R = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
import gzip

rows = list(csv.reader((gzip.open(path, "rt") if path.endswith(".gz") else open(path))))[2:]
zones = defaultdict(lambda: defaultdict(list))  # chip -> key -> [(s, e)]
open_ = {}
for r in rows:
    if not r[10].strip().endswith("-KERNEL"):
        continue
    chip, key = int(r[0]), (r[1], r[2], r[3])
    t = int(r[5])
    if r[11].strip() == "ZONE_START":
        open_[(chip, key)] = t
    elif (chip, key) in open_:
        zones[chip][key].append((open_.pop((chip, key)), t))
intra, edge, spans, starts_spread = [], [], [], []
for chip in sorted(zones):
    kz = zones[chip]
    mx = max(len(v) for v in kz.values())
    keys = [k for k, v in kz.items() if len(v) == mx]
    n = K * R
    progs = []
    for i in range(mx - n - 1, mx):
        zs = [sorted(kz[k])[i] for k in keys]
        progs.append((min(a for a, _ in zs), max(b for _, b in zs), max(a for a, _ in zs)))
    out = []
    for i in range(1, len(progs)):
        g = (progs[i][0] - progs[i - 1][1]) / F
        sp = (progs[i][1] - progs[i][0]) / F
        is_edge = (i - 1) % K == 0
        if i > 1:
            (edge if is_edge else intra).append(g)
        spans.append(sp)
        starts_spread.append((progs[i][2] - progs[i][0]) / F)
        out.append(("|" if is_edge else "") + f"(+{g:.2f}){sp:.2f}")
    print(f"chip {chip} [{len(keys)} risc]: " + " ".join(out))
med = lambda a: sorted(a)[len(a) // 2] if a else float("nan")
print(
    f"MEDIAN span {med(spans):.2f} | replay-edge gap {med(edge):.2f} (n={len(edge)}) | intra-trace gap "
    f"{med(intra):.2f} (n={len(intra)}) | start spread {med(starts_spread):.2f}"
)
