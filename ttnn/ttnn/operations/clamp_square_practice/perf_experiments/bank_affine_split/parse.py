# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Parse one --profile report dir: per-op DEVICE KERNEL DURATION + per-core zone spans (us).

usage: python parse.py <report_dir> [label1,label2,...] [--heat ZONE]
Ops are listed in execution order; labels (test order) are zipped onto GenericOp rows.
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MHZ = 1350.0


def ops(rd):
    f = next(Path(rd).glob("ops_perf_results*.csv"))
    rows = list(csv.DictReader(open(f)))
    return [r for r in rows if "Generic" in r["OP CODE"] or "generic" in r["OP CODE"].lower()]


def zones(rd):
    """{run_host_id: {zone: {(x,y): span_us}}}"""
    f = next(Path(rd).rglob("profile_log_device.csv"))
    lines = open(f).read().splitlines()
    hdr_i = next(i for i, l in enumerate(lines) if "core_x" in l)
    rdr = csv.DictReader(lines[hdr_i:], skipinitialspace=True)
    st = {}
    out = defaultdict(lambda: defaultdict(dict))
    for r in rdr:
        z = r.get("zone name", "").strip()
        if z not in ("reader_read", "writer_write", "writer_final_barrier", "compute_chain"):
            continue
        key = (r["run host ID"].strip(), z, int(r["core_x"]), int(r["core_y"]), r["RISC processor type"].strip())
        t = int(r["time[cycles since reset]"])
        if r["type"].strip() == "ZONE_START":
            st[key] = t
        elif key in st:
            out[key[0]][z][(key[2], key[3])] = (t - st.pop(key)) / MHZ
    return out


def heat(spans):
    xs = sorted({x for x, _ in spans})
    ys = sorted({y for _, y in spans})
    s = "y\\x " + " ".join(f"{x:4d}" for x in xs) + "\n"
    for y in ys:
        s += f"{y:3d} " + " ".join(f"{spans.get((x, y), 0):4.0f}" for x in xs) + "\n"
    return s


if __name__ == "__main__":
    rd = sys.argv[1]
    labels = sys.argv[2].split(",") if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    hz = sys.argv[sys.argv.index("--heat") + 1] if "--heat" in sys.argv else None
    o = ops(rd)
    zz = zones(rd)
    for i, r in enumerate(o):
        lab = labels[i] if labels and i < len(labels) else f"op{i}"
        rid = r["GLOBAL CALL COUNT"]
        z = zz.get(rid, {})
        ww = z.get("writer_write", {})
        fb = z.get("writer_final_barrier", {})
        tot = {c: ww[c] + fb.get(c, 0) for c in ww}
        rr = z.get("reader_read", {})
        line = f"{lab:20s} dur_ns={int(r['DEVICE KERNEL DURATION [ns]']):8d}"
        if tot:
            v = sorted(tot.values())
            line += f"  wr p50={statistics.median(v):6.1f} max={v[-1]:6.1f} min={v[0]:6.1f}"
        if rr:
            v = sorted(rr.values())
            line += f"  rd p50={statistics.median(v):6.1f} max={v[-1]:6.1f}"
        print(line)
        if hz and (hz == "all" or hz == lab):
            print(heat(rr if "--rd" in sys.argv else tot))
