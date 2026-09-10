"""Per-stage device zones, one variant per RUN -- WHERE does the fold's gain go?

The wall on the focus shape is roofline-gated (the round's peel: payload together = 473
GB/s, with a 14,225 ns ADDITIVE TRISC floor), so a TRISC-work deletion can be real and
still not move the wall.  This measures the DELETION directly: `compute_square` and
`compute_reduce` occupancy per core, base vs candidate.

  RMS_ZVARIANT=base|grp8|flatinf  RMS_ZCASE=P04_8192x1024
  scripts/tt-probe.sh rms_norm_ttnn < zones_fold.py

Occupancy caveat (references/device-zone-scope-attribution.md): `compute_square` wraps its
own cb_wait_front, so its number is WAIT + WORK.  `compute_reduce` does not wait on
cb_x_squared's producer (same thread), so it is work.
"""

import collections
import csv
import importlib.util
import os
import statistics
import sys
from pathlib import Path

os.environ["RMS_STAGE_ZONES"] = "1"
os.environ["RMS_NO_MAIN"] = "1"
for k, v in (
    ("TT_METAL_DEVICE_PROFILER", "1"),
    ("TT_METAL_PROFILER_MID_RUN_DUMP", "1"),
    ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
    ("TT_METAL_LOGGER_LEVEL", "error"),
):
    os.environ.setdefault(k, v)

HERE = Path(
    os.environ.get("RMS_EXP_DIR")
    or "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal"
    "/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/square_fold_ceiling"
)

spec = importlib.util.spec_from_file_location("bench_fold", HERE / "bench_fold.py")
B = importlib.util.module_from_spec(spec)
sys.modules["bench_fold"] = B
spec.loader.exec_module(B)

import ttnn  # noqa: E402

CSVP = Path("generated/profiler/.logs/profile_log_device.csv")
FREQ = 1.35  # GHz -> cycles to ns


def parse_zones():
    rows = []
    with CSVP.open() as fh:
        fh.readline()
        rdr = csv.reader(fh)
        header = [h.strip() for h in next(rdr)]
        idx = {h: i for i, h in enumerate(header)}
        for r in rdr:
            if len(r) >= len(header):
                rows.append(r)
    ci = {
        k: idx[k]
        for k in (
            "core_x",
            "core_y",
            "RISC processor type",
            "time[cycles since reset]",
            "zone name",
            "type",
            "run host ID",
        )
    }
    target = sorted({int(r[ci["run host ID"]]) for r in rows})[-1]
    open_stack = collections.defaultdict(list)
    per_core = collections.defaultdict(lambda: collections.defaultdict(float))
    counts = collections.defaultdict(int)
    span = collections.defaultdict(lambda: [None, None])
    for r in rows:
        if int(r[ci["run host ID"]]) != target:
            continue
        key = (r[ci["core_x"]], r[ci["core_y"]], r[ci["RISC processor type"]])
        zone = r[ci["zone name"]].strip()
        t = int(r[ci["time[cycles since reset]"]])
        lo, hi = span[key]
        span[key] = [t if lo is None else min(lo, t), t if hi is None else max(hi, t)]
        if r[ci["type"]].strip() == "ZONE_START":
            open_stack[(key, zone)].append(t)
        elif r[ci["type"]].strip() == "ZONE_END" and open_stack[(key, zone)]:
            dt = t - open_stack[(key, zone)].pop()
            per_core[(zone, r[ci["RISC processor type"]])][key] += dt
            counts[(zone, r[ci["RISC processor type"]])] += 1
    return per_core, counts, target, span


def main():
    label = os.environ.get("RMS_ZVARIANT", "base")
    case = os.environ.get("RMS_ZCASE", B.FOCUS)
    if CSVP.exists():
        CSVP.unlink()
    device = ttnn.open_device(device_id=0)
    try:
        B._reset()
        B.VARIANTS[label]()
        run, _expected, live = B.build(device, case)
        out = run()
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
        for t in live:
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
    finally:
        B._reset()
        ttnn.close_device(device)
    per_core, counts, target, span = parse_zones()
    print(f"RESULT ---- zones variant={label} case={case} (run {target}) ----")
    print(f"RESULT {'zone':26s} {'risc':8s} {'occ/core ns':>12s} {'max/core':>10s} {'n/core':>8s} {'cores':>6s}")
    out = []
    for k, cores in per_core.items():
        vals = list(cores.values())
        out.append((statistics.mean(vals), k[0], k[1], max(vals), counts[k] / len(vals), len(vals)))
    for mean, z, ri, mx, n, nc in sorted(out, reverse=True):
        print(f"RESULT {label:9s} {z:26s} {ri:8s} {mean / FREQ:12.1f} {mx / FREQ:10.1f} {n:8.2f} {nc:6d}")
    print("RESULT ---- per-RISC total marker span (kernel coverage check) ----")
    byrisc = collections.defaultdict(list)
    for (cx, cy, ri), (lo, hi) in span.items():
        byrisc[ri].append((hi - lo) / FREQ)
    for ri, v in sorted(byrisc.items()):
        print(f"RESULT {label:9s} span {ri:8s} mean={statistics.mean(v):10.1f} max={max(v):10.1f} n={len(v)}")


main()
