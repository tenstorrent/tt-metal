# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-DISPATCH per-stage zone statistics from a `--profile` report.

`perf_experiments/parse_zones.py` aggregates the whole device log, which is exactly wrong for a
bake-off: one pytest process dispatches the op once PER VARIANT, so the log holds N interleaved
variants. This splits them on the log's `run host ID` column (monotone per dispatch), so dispatch k
maps to the k-th entry of MOE_HEAD_VARIANTS.

    python3 ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/reader_head_scheduling/parse_head_zones.py \
        [report_dir] [--labels baseline,x_trid,...]

A stage's duration is END - BEGIN on the same (core, risc, zone), matched in arrival order (a zone
never nests inside itself here — one `{ }` block per stage).
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

REPORTS = Path("generated/profiler/reports")


def newest_report(explicit=None):
    if explicit:
        return Path(explicit)
    dirs = sorted((d for d in REPORTS.iterdir() if d.is_dir()), key=lambda d: d.name)
    for d in reversed(dirs):
        if (d / "profile_log_device.csv").exists():
            return d
    raise SystemExit(f"no report with profile_log_device.csv under {REPORTS}")


def whole_op(report):
    out = []
    for csv_path in sorted(report.glob("ops_perf_results*.csv")):
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                if r.get("OP CODE") != "GenericOpDeviceOperation":
                    continue
                out.append(int(float(r["DEVICE KERNEL DURATION [ns]"])))
    return out


def zones_by_dispatch(report):
    """{run_host_id: {zone: [durations_ns]}} — one entry per dispatch."""
    path = report / "profile_log_device.csv"
    freq = None
    with open(path) as fh:
        header = fh.readline()
        for tok in header.split(","):
            if "CHIP_FREQ" in tok:
                pass
        # "ARCH: blackhole, CHIP_FREQ[MHz]: 1350, ..."
        for part in header.split(","):
            if "CHIP_FREQ" in part:
                freq = float(part.split(":")[1])
        cols = [c.strip() for c in fh.readline().split(",")]
        rows = list(csv.reader(fh))
    assert freq, "CHIP_FREQ missing from the device-log header"
    idx = {name: i for i, name in enumerate(cols)}
    ci, cx, cy = idx["core_x"], idx["core_y"], idx["RISC processor type"]
    ct, cz, crun = idx["time[cycles since reset]"], idx["zone name"], idx["run host ID"]
    ctype = idx["type"]

    open_begin = defaultdict(list)
    per_dispatch = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if len(r) <= max(ci, cx, cy, ct, cz, crun, ctype):
            continue
        zone = r[cz].strip()
        if not zone or not zone.startswith(("reader_", "writer_", "compute_")):
            continue
        key = (r[ci].strip(), r[cx].strip(), r[cy].strip(), zone, r[crun].strip())
        kind = r[ctype].strip().upper()
        cycles = int(r[ct])
        if kind.startswith("ZONE_START") or kind == "BEGIN":
            open_begin[key].append(cycles)
        elif kind.startswith("ZONE_END") or kind == "END":
            if open_begin[key]:
                start = open_begin[key].pop(0)
                per_dispatch[r[crun].strip()][zone].append((cycles - start) * 1000.0 / freq)
    return per_dispatch


def main():
    args = [a for a in sys.argv[1:]]
    labels = []
    if "--labels" in args:
        i = args.index("--labels")
        labels = args[i + 1].split(",")
        del args[i : i + 2]
    report = newest_report(args[0] if args else None)
    print(f"report: {report}")
    print(f"whole-op DEVICE KERNEL DURATION per dispatch: {whole_op(report)}")
    per = zones_by_dispatch(report)
    for k, (run_id, zones) in enumerate(sorted(per.items(), key=lambda kv: int(kv[0]))):
        label = labels[k] if k < len(labels) else f"dispatch{k}"
        print(f"\n=== dispatch {k} (run host ID {run_id}) — {label} ===")
        for zone in sorted(zones, key=lambda z: -sum(zones[z]) / max(1, len(zones[z]))):
            d = zones[zone]
            print(f"  {zone:<24} n={len(d):<5} mean={sum(d) / len(d):>9.0f} ns  max={max(d):>9.0f} ns")


if __name__ == "__main__":
    main()
