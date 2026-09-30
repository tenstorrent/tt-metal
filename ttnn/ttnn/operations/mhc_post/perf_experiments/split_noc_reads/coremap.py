"""Per-core kernel wall (max over RISCs of the *-KERNEL zone, us) as a physical-grid map, for the runs matched to
run_order.jsonl (the last N run host IDs of profile_log_device.csv).
usage: OPS_CSV=<ops_perf_results.csv> python coremap.py <profile_log_device.csv> <variant>[,<variant>..] <shape> <mode> [zone]
(run host IDs = GLOBAL CALL COUNT of the last len(run_order) GenericOp rows of OPS_CSV)
With [zone] (e.g. dm_read_barrier), prints the per-core summed time of that zone instead."""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

MHZ = 1350.0
path, variants, shape, mode = sys.argv[1], sys.argv[2].split(","), sys.argv[3], sys.argv[4]
zone = sys.argv[5] if len(sys.argv) > 5 else None  # "zone" or "zone@RISC" (e.g. compute_mix@TRISC_1)
zrisc = None
if zone and "@" in zone:
    zone, zrisc = zone.split("@")
order = [json.loads(l) for l in open(Path(__file__).parent / "run_order.jsonl")]
kstart, kend = defaultdict(lambda: 1 << 62), defaultdict(int)
zacc, zst = defaultdict(float), {}
runs = set()
with open(path) as f:
    next(f)
    r = csv.reader(f)
    next(r)
    for row in r:
        row = [x.strip() for x in row]
        run = int(row[7])
        x, y, risc, t, z, ty = int(row[1]), int(row[2]), row[3], int(row[5]), row[10], row[11]
        runs.add(run)
        if z.endswith("-KERNEL"):
            if ty == "ZONE_START":
                kstart[(run, x, y)] = min(kstart[(run, x, y)], t)
            else:
                kend[(run, x, y)] = max(kend[(run, x, y)], t)
        elif zone and z == zone and (zrisc is None or risc == zrisc):
            k = (run, x, y, risc)
            if ty == "ZONE_START":
                zst[k] = t
            elif k in zst:
                zacc[(run, x, y)] += t - zst.pop(k)
import os

with open(os.environ["OPS_CSV"]) as f:
    ops = [r for r in csv.reader(f)][1:]
runs = [int(r[2]) for r in ops if r[0].startswith("GenericOp")][-len(order) :]
for v in variants:
    idx = [i for i, o in enumerate(order) if o["variant"] == v and o["shape"] == shape and o["mode"] == mode][-1]
    run = runs[idx]
    cores = {(x, y) for (rn, x, y) in kend if rn == run}
    xs, ys = sorted({c[0] for c in cores}), sorted({c[1] for c in cores})
    t0 = min(kstart[(run, x, y)] for x, y in cores)
    val = {c: ((zacc[(run,) + c] if zone else kend[(run,) + c] - t0) / MHZ) for c in cores}
    vs = sorted(val.values())
    print(
        f"== {v} {shape} {mode} run {run} {'zone ' + zone if zone else 'kernel end (us from first start)'}: "
        f"max {vs[-1]:.1f} p50 {vs[len(vs)//2]:.1f} min {vs[0]:.1f}"
    )
    print("  y\\x " + "".join(f"{x:6d}" for x in xs))
    for y in ys:
        print(f"  {y:3d} " + "".join(f"{val[(x, y)]:6.0f}" if (x, y) in val else "     ." for x in xs))
