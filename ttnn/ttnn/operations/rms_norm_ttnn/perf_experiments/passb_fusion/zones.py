"""passb_fusion — per-stage device zones for the baseline vs the fused pass B.

Attribution: how much does pass B itself cost in each variant, per compute
thread (UNPACK / MATH / PACK)?  The fusion trades an unpack+pack of the
intermediate for a per-face DEST->SrcA move + ZEROACC + MOP restart, so the
three thread numbers are the evidence for which side is bigger.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
os.environ["RMS_STAGE_ZONES"] = "1"

import collections
import csv
import statistics
from pathlib import Path

import ttnn

import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD

import bench  # same directory

HERE = Path(__file__).resolve().parent
CSV = Path("generated/profiler/.logs/profile_log_device.csv")
FREQ = 1.35  # cycles -> ns at 1350 MHz

PASS_B_ZONES = {
    "compute_scale",
    "compute_gamma_mul",
    "compute_bias_add",
    "compute_passb_fused",
}


def parse_zones():
    rows = []
    with CSV.open() as fh:
        fh.readline()
        rdr = csv.reader(fh)
        header = [h.strip() for h in next(rdr)]
        idx = {h: i for i, h in enumerate(header)}
        for r in rdr:
            if len(r) < len(header):
                continue
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
    runs = sorted({int(r[ci["run host ID"]]) for r in rows})
    target = runs[-1]
    open_stack = collections.defaultdict(list)
    agg = collections.defaultdict(list)
    for r in rows:
        if int(r[ci["run host ID"]]) != target:
            continue
        key = (r[ci["core_x"]], r[ci["core_y"]], r[ci["RISC processor type"]])
        zone = r[ci["zone name"]].strip()
        typ = r[ci["type"]].strip()
        t = int(r[ci["time[cycles since reset]"]])
        if typ == "ZONE_START":
            open_stack[(key, zone)].append(t)
        elif typ == "ZONE_END":
            st = open_stack[(key, zone)]
            if st:
                agg[(zone, r[ci["RISC processor type"]], key)].append(t - st.pop())
    return agg, target


def main():
    name = os.environ.get("RMS_CASE", "focus")
    variants = os.environ.get("RMS_VARIANTS", "base,fuse").split(",")
    saved = PD.KERNEL_DIR
    out = {}
    for v in variants:
        if CSV.exists():
            CSV.unlink()
        PD.KERNEL_DIR = HERE / f"k_{v}"
        device = ttnn.open_device(device_id=0)
        try:
            run, expected, base_ns, live = bench.build(device, name)
            run()
            ttnn.synchronize_device(device)
            bench._read_kernel_ns(device)
            run()
            ttnn.synchronize_device(device)
            ns = bench._read_kernel_ns(device)
        finally:
            PD.KERNEL_DIR = saved
            ttnn.close_device(device)
        agg, target = parse_zones()
        out[v] = (ns, agg)

    for v in variants:
        ns, agg = out[v]
        print(f"RESULT ===== variant={v} case={name} kernel_ns={ns:.0f} =====")
        # per (zone, risc): total ns per core, then report the MAX over cores
        per = collections.defaultdict(list)
        for (zone, risc, key), vals in agg.items():
            per[(zone, risc)].append((sum(vals), len(vals)))
        rowsout = []
        for (zone, risc), vals in per.items():
            tot = max(x[0] for x in vals) / FREQ
            n = max(x[1] for x in vals)
            mark = "  <== PASS B" if zone in PASS_B_ZONES else ""
            rowsout.append((tot, zone, risc, n, mark))
        for tot, zone, risc, n, mark in sorted(rowsout, reverse=True):
            print(f"RESULT   {zone:26s} {risc:8s} n={n:3d} max_core_total_ns={tot:9.1f}{mark}")
        pb = collections.defaultdict(float)
        for (zone, risc, key), vals in agg.items():
            if zone in PASS_B_ZONES:
                pb[(risc, key)] += sum(vals)
        byrisc = collections.defaultdict(list)
        for (risc, key), s in pb.items():
            byrisc[risc].append(s / FREQ)
        for risc in sorted(byrisc):
            v_ = byrisc[risc]
            print(
                f"RESULT   PASS-B TOTAL {risc:8s} ncores={len(v_):3d} "
                f"max={max(v_):9.1f} mean={statistics.mean(v_):9.1f}"
            )
