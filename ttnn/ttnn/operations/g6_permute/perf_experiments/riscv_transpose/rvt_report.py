#!/usr/bin/env python3
"""rvt_report.py - numbers of the riscv_transpose bake-off from one profiled pytest run.

  rvt_report.py micro LOG DEVCSV           per variant: median cycles of zone RVT-UNIT per RISC lane kind
  rvt_report.py op LOG OPSCSV [DEVCSV] [--zones]
                                           per variant: median DEVICE KERNEL DURATION of launches 2..N (OPSCSV '-':
                                           kernel spans of DEVCSV via zone_table.py), --zones: per-RISC zone totals
LOG is the pytest stdout (lines 'RVT_MICRO <id> first_op=A last_op=B' / 'RVT_OP <id> first_op=A last_op=B');
OPSCSV is ops_perf_results_*.csv, DEVCSV is .logs/profile_log_device.csv of the same run.
Standard library only.
"""

import csv
import re
import statistics
import sys
from collections import defaultdict


def variants(log_path, tag):
    """Last occurrence per id (a precompile collect pass prints the same lines first, with no ops)."""
    out = {}
    for line in open(log_path, errors="replace"):
        m = re.search(rf"{tag} (\S+) first_op=(\d+) last_op=(\d+)", line)
        if m:
            out[m.group(1)] = (m.group(1), int(m.group(2)), int(m.group(3)))
    return list(out.values())


def device_zones(dev_csv):
    """{op_id: {(core, risc, zone): [cycles...]}}"""
    zones = defaultdict(lambda: defaultdict(list))
    opened = {}
    with open(dev_csv) as f:
        next(f)
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        ix = {h: i for i, h in enumerate(header)}
        for row in reader:
            if len(row) < len(header):
                continue
            kind = row[ix["type"]].strip()
            if kind not in ("ZONE_START", "ZONE_END"):
                continue
            name = row[ix["zone name"]].strip()
            run = int(row[ix["run host ID"]])
            core = (int(row[ix["core_x"]]), int(row[ix["core_y"]]))
            risc = row[ix["RISC processor type"]].strip()
            t = int(row[ix["time[cycles since reset]"]])
            key = (run, core, risc, name)
            if kind == "ZONE_START":
                opened[key] = t
            elif key in opened:
                zones[run >> 10][(core, risc, name)].append(t - opened.pop(key))
    return zones


def micro(log_path, dev_csv):
    zones = device_zones(dev_csv)
    for vid, first, last in variants(log_path, "RVT_MICRO"):
        per_risc = defaultdict(list)
        for op in range(first + 1, last + 1):  # launch 1 carries the cold instruction cache
            for (core, risc, name), cyc in zones.get(op, {}).items():
                if name == "RVT-UNIT":
                    per_risc[risc].extend(cyc[1:])  # first unit of a launch: cold data cache
        cells = "  ".join(f"{r}={statistics.median(c):.0f}" for r, c in sorted(per_risc.items()))
        print(f"{vid:18s} {cells}")


def span_durations(dev_csv):
    """{op_id: kernel span ns} from the device log via zone_table.py --brief (= DEVICE KERNEL DURATION)."""
    import subprocess
    from pathlib import Path

    zt = Path(__file__).resolve().parents[6] / ".claude/skills/tooling/scripts/zone_table.py"
    text = subprocess.run([sys.executable, str(zt), dev_csv, "--brief"], capture_output=True, text=True).stdout
    out = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) > 6 and f[0].isdigit() and f[2].isdigit():
            out[int(f[2])] = float(f[6].replace(",", ""))
    return out


def op_durations(ops_csv):
    if ops_csv == "-":
        return None
    out = {}
    with open(ops_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                op = int(row["GLOBAL CALL COUNT"])
                out[op] = float(row["DEVICE KERNEL DURATION [ns]"])
            except (KeyError, ValueError):
                continue
    return out


def op(log_path, ops_csv, dev_csv=None):
    durs = op_durations(ops_csv)
    if durs is None:
        durs = span_durations(dev_csv)
    zones = device_zones(dev_csv) if dev_csv and "--zones" in sys.argv else {}
    for vid, first, last in variants(log_path, "RVT_OP"):
        ds = [durs[o] for o in range(first + 1, last + 1) if o in durs]
        if not ds:
            print(f"{vid:28s} no durations (ops {first}..{last})")
            continue
        line = f"{vid:28s} median={statistics.median(ds):9.0f} ns  min={min(ds):9.0f}  max={max(ds):9.0f}  n={len(ds)}"
        print(line)
        if zones:
            # per (risc, zone): median over launches 2..N of the per-lane total, max over cores
            per = defaultdict(list)
            for o in range(first + 1, last + 1):
                lane_tot = defaultdict(lambda: defaultdict(int))
                for (core, risc, name), cyc in zones.get(o, {}).items():
                    lane_tot[(risc, name)][core] += sum(cyc)
                for k, by_core in lane_tot.items():
                    per[k].append((max(by_core.values()), statistics.median(by_core.values())))
            for risc, name in sorted(per):
                vals = per[(risc, name)]
                mx = statistics.median(v[0] for v in vals)
                md = statistics.median(v[1] for v in vals)
                print(f"    {risc:8s} {name:28s} max-core {mx:8.0f} cyc  median-core {md:8.0f} cyc")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "micro":
        micro(sys.argv[2], sys.argv[3])
    elif mode == "op":
        op(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)  # OPSCSV '-': spans of DEVCSV
    else:
        sys.exit(__doc__)
