# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Mechanism check for `reader_boot_order`: a per-stage zone timeline on the FOCUS
# shape, one variant per invocation (RMS_VARIANT).  Confirms WHERE the ~1.2 us
# went -- i.e. that `reader_native_publish` really does land at ~600 ns and that
# `reader_read_gamma` really does run UNDER pass A rather than in front of it.
#
# Run through the device wrapper:
#   RMS_VARIANT=d_pub_first_split scripts/tt-probe.sh rms_norm_ttnn <<'PYEOF'
#   exec(open(".../zones_boot_order.py").read())
#   PYEOF
import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
os.environ["RMS_STAGE_ZONES"] = "1"

import collections
import csv
import statistics
import sys
from pathlib import Path

HERE = Path("ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/reader_boot_order").resolve()
sys.path.insert(0, str(HERE))

import ttnn
import bench_boot_order as B
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD

CSVP = Path("generated/profiler/.logs/profile_log_device.csv")
FREQ = 1.35  # cycles -> ns at 1350 MHz
VARIANT = os.environ.get("RMS_VARIANT", "base")
CASE = os.environ.get("RMS_CASE", "F_w7168_28c")


def parse():
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
    per_core_start = {}
    events = []
    for r in rows:
        if int(r[ci["run host ID"]]) != target:
            continue
        core = (r[ci["core_x"]], r[ci["core_y"]])
        t = int(r[ci["time[cycles since reset]"]])
        per_core_start[core] = min(per_core_start.get(core, t), t)
        events.append((core, r[ci["RISC processor type"]], r[ci["zone name"]].strip(), r[ci["type"]].strip(), t))
    # zone -> list of (rel_start_ns, rel_end_ns) relative to that core's own first marker
    openst = collections.defaultdict(list)
    spans = collections.defaultdict(list)
    for core, risc, zone, typ, t in events:
        if typ == "ZONE_START":
            openst[(core, risc, zone)].append(t)
        elif typ == "ZONE_END":
            st = openst[(core, risc, zone)]
            if st:
                s = st.pop()
                base = per_core_start[core]
                spans[(zone, risc)].append(((s - base) / FREQ, (t - base) / FREQ))
    return spans, target


device = ttnn.open_device(device_id=0)
shipped = PD.KERNEL_DIR
try:
    if B.VARIANTS[VARIANT] is not None:
        PD.KERNEL_DIR = B.VARIANTS[VARIANT]
    run, expected, ceiling, live = B.build(device, CASE)
    run()
    ttnn.synchronize_device(device)
    B._read_kernel_ns(device)
    if CSVP.exists():
        CSVP.unlink()
    run()
    ttnn.synchronize_device(device)
    ns = B._read_kernel_ns(device)
    print(f"RESULT variant={VARIANT} case={CASE} ns={ns}")
finally:
    PD.KERNEL_DIR = shipped
    ttnn.close_device(device)

spans, target = parse()
print(f"RESULT ---- zone timeline, ns from each core's OWN first marker (run {target}) ----")
print(f"RESULT {'zone':28s}{'risc':8s}{'n':>4s}{'start_med':>10s}{'end_med':>9s}{'dur_med':>9s}{'end_spread':>11s}")
out = []
for (zone, risc), v in spans.items():
    starts = [s for s, e in v]
    ends = [e for s, e in v]
    durs = [e - s for s, e in v]
    out.append(
        (
            statistics.median(starts),
            zone,
            risc,
            len(v),
            statistics.median(starts),
            statistics.median(ends),
            statistics.median(durs),
            max(ends) - min(ends),
        )
    )
for _, zone, risc, n, s, e, d, spread in sorted(out):
    print(f"RESULT {zone:28s}{risc:8s}{n:4d}{s:10.0f}{e:9.0f}{d:9.0f}{spread:11.0f}")
