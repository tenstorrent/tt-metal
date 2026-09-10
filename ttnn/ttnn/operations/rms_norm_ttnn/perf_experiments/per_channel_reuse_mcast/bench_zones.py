# Perf experiment `per_channel_reuse_mcast` -- where does the multicast's time go?
#
# Runs ONE case under ONE variant with RMS_STAGE_ZONES=1 and aggregates the
# reader's per-stage device zones, so `reader_read_gamma` + `reader_read_bias`
# (baseline) can be compared against `reader_pc_inject` / `reader_pc_recv`
# (candidate) rather than guessed at.
import collections
import csv
import importlib
import importlib.util
import os
import statistics
import sys
from pathlib import Path

os.environ["RMS_STAGE_ZONES"] = "1"
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "tests/ttnn/unit_tests/operations/rms_norm_ttnn/probes"))

import bench_r3  # noqa: E402
import ttnn  # noqa: E402
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD  # noqa: E402

OP = importlib.import_module("ttnn.operations.rms_norm_ttnn.rms_norm_ttnn")
CSVP = Path("generated/profiler/.logs/profile_log_device.csv")
FREQ = 1.35


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PDM = _load("pd_mcast", HERE / "pd_mcast.py")
SHIPPED = OP.create_program_descriptor


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
    agg = collections.defaultdict(list)
    for r in rows:
        if int(r[ci["run host ID"]]) != target:
            continue
        key = (r[ci["core_x"]], r[ci["core_y"]], r[ci["RISC processor type"]])
        zone = r[ci["zone name"]].strip()
        t = int(r[ci["time[cycles since reset]"]])
        if r[ci["type"]].strip() == "ZONE_START":
            open_stack[(key, zone)].append(t)
        elif r[ci["type"]].strip() == "ZONE_END" and open_stack[(key, zone)]:
            agg[(zone, r[ci["RISC processor type"]])].append(t - open_stack[(key, zone)].pop())
    return agg, target


def main():
    case = os.environ.get("RMS_CASES", "P2_int1024_gb")
    variant = os.environ.get("RMS_VARIANTS", "mcast")
    if variant == "base":
        OP.create_program_descriptor = SHIPPED
        PD.KERNEL_DIR = HERE / "k_base"
    else:
        OP.create_program_descriptor = PDM.create_program_descriptor
        PDM.PC_MCAST_ENABLE = variant != "pd_off"
        PDM.KERNEL_DIR = HERE / "k_mcast"
    if CSVP.exists():
        CSVP.unlink()
    device = ttnn.open_device(device_id=0)
    try:
        ns, p, r = bench_r3.measure(device, case)
        print(f"RESULT.zone case={case} variant={variant} ns={ns:.0f} pcc={p:.6f} relrms={r:.3e}", flush=True)
    finally:
        ttnn.close_device(device)
    agg, target = parse_zones()
    print(f"RESULT ---- zones ({case} / {variant}, run {target}) ----")
    print(f"RESULT {'zone':26s} {'risc':8s} {'n':>5s} {'mean_ns':>9s} {'max_ns':>9s}")
    out = sorted(((statistics.mean(v), z, ri, len(v), max(v)) for (z, ri), v in agg.items()), reverse=True)
    for mean, z, ri, n, mx in out:
        print(f"RESULT {z:26s} {ri:8s} {n:5d} {mean/FREQ:9.1f} {mx/FREQ:9.1f}")


main()
