"""slow_core.py <profile_log_device.csv>: per op run, slowest core end and its per-RISC end + reader_affine total (us)."""
import csv
import sys
from collections import defaultdict

with open(sys.argv[1]) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
open_, zones = {}, defaultdict(list)
for r in rows[1:]:
    r = [c.strip() for c in r]
    k = (
        r[idx["run host ID"]],
        (r[idx["core_x"]], r[idx["core_y"]]),
        r[idx["RISC processor type"]],
        r[idx["zone name"]],
    )
    t = int(r[idx["time[cycles since reset]"]])
    if r[idx["type"]] == "ZONE_START":
        open_[k] = t
    elif k in open_:
        zones[k[0]].append((k[1], k[2], k[3], open_.pop(k), t))
for run, ev in zones.items():
    if not any(z.startswith("reader_") for _, _, z, _, _ in ev):
        continue
    t0 = min(s for _, _, z, s, _ in ev if z.endswith("-KERNEL"))
    end = defaultdict(int)
    for core, risc, z, s, t in ev:
        if z.endswith("-KERNEL"):
            end[core] = max(end[core], t - t0)
    slow = max(end, key=end.get)
    us = lambda c: c / freq
    risc_end = {risc: us(t - t0) for core, risc, z, s, t in ev if core == slow and z.endswith("-KERNEL")}
    aff = sum(us(t - s) for core, risc, z, s, t in ev if core == slow and z == "reader_affine")
    heavy_aff = sorted(us(t - s) for core, risc, z, s, t in ev if z == "reader_affine")
    print(
        f"run {run} slow {slow} end {us(end[slow]):.2f} "
        + " ".join(f"{k}={v:.2f}" for k, v in sorted(risc_end.items()))
        + f" reader_affine(slow)={aff:.2f} reader_affine max={heavy_aff[-1]:.2f} p50={heavy_aff[len(heavy_aff)//2]:.2f}"
    )
