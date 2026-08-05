# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage zone attribution for BENCH B, so the win is split into fold vs TRANSPORT vs sync.

    python3 <this dir>/read_zones.py <profile_log_device.csv> <ops_perf_results.csv> [config_id ...]

`MaybeDeviceZoneScope` names every stage of both bench-B kernels, and the device log records a
ZONE_START/ZONE_END cycle pair per stage per core per round.  This script sums them per launch on
(a) the ROOT core -- the one that runs `compute_root_fused` -- and (b) a representative MEMBER, and
prints ns totals at the box clock, PER RISC (a compute zone is recorded independently on TRISC_0/1/2,
so merging them triples the call count -- observed while bringing this script up).  Launch order is the `run host ID` order, which is the same
order transport_manifest.jsonl records.

Deliberately NOT importable as a package module -- run it directly.
"""
import collections
import csv
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
CLOCK_MHZ = 1350.0  # blackhole p150b; device->get_clock_rate_mhz()

ROOT_MARK = "compute_root_fused"


def main():
    dev_csv, ops_csv = sys.argv[1], sys.argv[2]
    wanted = set(sys.argv[3:])
    call_ids = [
        int(r["GLOBAL CALL COUNT"]) for r in csv.DictReader(open(ops_csv)) if r["OP CODE"] == "GenericOpDeviceOperation"
    ]
    man = [json.loads(line) for line in open(HERE / "transport_manifest.jsonl")]
    launched = [m for m in man if not m.get("l1_oom")]

    # (launch_id, core, zone) -> [total_cycles, count]; pair START with the next END.
    acc = collections.defaultdict(lambda: [0, 0])
    open_at = {}
    ids = set()
    with open(dev_csv) as f:
        f.readline()
        rdr = csv.reader(f)
        hdr = [h.strip() for h in next(rdr)]
        ci = {
            k: hdr.index(k)
            for k in (
                "core_x",
                "core_y",
                "RISC processor type",
                "time[cycles since reset]",
                "run host ID",
                "zone name",
                "type",
            )
        }
        for r in rdr:
            z = r[ci["zone name"]].strip()
            if z.endswith("-FW") or z.endswith("-KERNEL"):
                continue
            rid = int(r[ci["run host ID"]])
            ids.add(rid)
            core = (r[ci["core_x"]].strip(), r[ci["core_y"]].strip())
            risc = r[ci["RISC processor type"]].strip()
            t = int(r[ci["time[cycles since reset]"]])
            key = (rid, core, risc, z)
            if r[ci["type"]].strip() == "ZONE_START":
                open_at[key] = t
            else:
                s = open_at.pop(key, None)
                if s is not None:
                    acc[key][0] += t - s
                    acc[key][1] += 1

    # The ops CSV's GenericOp rows line up 1:1 with the manifest's LAUNCHING rows, warm-up
    # included -- verified by the flat variant showing a non-zero `writer_gather_zero` and no
    # compact-only zone.  (An off-by-one here silently swaps two variants' numbers.)
    for idx, rid in enumerate(call_ids):
        if idx >= len(launched):
            break
        m = launched[idx]
        if wanted and m["config"] not in wanted:
            continue
        cores = {c for (r_, c, _rr, _z) in acc if r_ == rid}
        roots = {c for (r_, c, _rr, z) in acc if r_ == rid and z == ROOT_MARK}
        root = sorted(roots)[0] if roots else None
        member = sorted(cores - roots)[0] if (cores - roots) else None
        print(
            f"\n== {m['config']:18s} {m['variant']:14s} rounds={m['num_blocks']} "
            f"BLOCK_ROWS={m['block_rows']} G={m['group_size']}   ns={m.get('ns','?')}"
        )
        for label, core in (("ROOT", root), ("MEMBER", member)):
            if core is None:
                continue
            for risc in ("BRISC", "TRISC_0"):
                zs = sorted(
                    (
                        (z, v[0] * 1000.0 / CLOCK_MHZ, v[1])
                        for (r_, c, rr, z), v in acc.items()
                        if r_ == rid and c == core and rr == risc
                    ),
                    key=lambda t: -t[1],
                )
                if not zs:
                    continue
                tot = sum(v for _z, v, _n in zs)
                print(f"   {label} {risc:7s} core {core}  zones sum {tot:8.0f} ns")
                for z, v, n in zs:
                    print(f"      {z:24s} {v:8.0f} ns  n={n:3d}  {v/max(n,1):7.0f} ns/call")


if __name__ == "__main__":
    main()
