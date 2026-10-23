"""Per-zone timeline report from generated/profiler/.logs/profile_log_device.csv.

usage: python zone_report.py [csv] [--runs N_LAST] [--dev D]
For each (device, run host id) of the last N runs: t0 = earliest *-KERNEL start on that chip for that run.
Per zone (+ RISC): instances, cores, per-core summed duration (p50 / max, us), first start / last end offset (us).
Cycles -> us at the CHIP_FREQ in the header.
"""
import csv, sys, collections, statistics, argparse

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?", default="generated/profiler/.logs/profile_log_device.csv")
ap.add_argument("--runs", type=int, default=2)
ap.add_argument("--dev", type=int, default=None)
ap.add_argument("--only", type=int, default=None, help="only this run host id")
a = ap.parse_args()

with open(a.csv) as fh:
    hdr = fh.readline()
    mhz = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rd = csv.reader(fh)
    next(rd)
    rows = [r for r in rd if len(r) > 12]

by = collections.defaultdict(list)
for r in rows:
    dev, x, y, risc, _, t, _, run = r[:8]
    name, typ = r[10].strip(), r[11].strip()
    by[(int(dev), int(run))].append((int(x), int(y), risc.strip(), name, typ, int(t)))

runs_per_dev = collections.defaultdict(list)
for d, run in by:
    runs_per_dev[d].append(run)

us = lambda c: c / mhz
for d in sorted(runs_per_dev):
    if a.dev is not None and d != a.dev:
        continue
    for run in sorted(runs_per_dev[d])[-a.runs :]:
        if a.only is not None and run != a.only:
            continue
        ev = by[(d, run)]
        ks = [t for (_, _, _, n, ty, t) in ev if n.endswith("-KERNEL") and ty == "ZONE_START"]
        ke = [t for (_, _, _, n, ty, t) in ev if n.endswith("-KERNEL") and ty == "ZONE_END"]
        t0 = min(ks)
        print(f"=== dev {d} run {run}: kernel span {us(max(ke) - t0):.2f} us")
        open_ = {}
        per = collections.defaultdict(lambda: collections.defaultdict(float))  # (zone,risc) -> core -> dur
        cnt = collections.Counter()
        first = {}
        last = {}
        for x, y, risc, n, ty, t in sorted(ev, key=lambda e: e[5]):
            k = (x, y, risc, n)
            if ty == "ZONE_START":
                open_.setdefault(k, []).append(t)
            elif ty == "ZONE_END" and open_.get(k):
                s = open_[k].pop()
                zk = (n, risc)
                per[zk][(x, y)] += t - s
                cnt[zk] += 1
                first[zk] = min(first.get(zk, s), s)
                last[zk] = max(last.get(zk, t), t)
        print(f"{'zone':22s}{'risc':8s}{'inst':>6s}{'cores':>6s}{'p50':>8s}{'max':>8s}{'first':>8s}{'last':>8s}")
        for zk in sorted(per, key=lambda z: first[z]):
            v = sorted(per[zk].values())
            if zk[0].endswith("-FW"):
                continue
            print(
                f"{zk[0]:22s}{zk[1]:8s}{cnt[zk]:6d}{len(v):6d}{us(statistics.median(v)):8.2f}{us(v[-1]):8.2f}"
                f"{us(first[zk] - t0):8.2f}{us(last[zk] - t0):8.2f}"
            )
            print(
                f"{zk[0]:22s}{zk[1]:8s}{cnt[zk]:6d}{len(v):6d}{us(statistics.median(v)):8.2f}{us(v[-1]):8.2f}"
                f"{us(first[zk] - t0):8.2f}{us(last[zk] - t0):8.2f}"
            )
