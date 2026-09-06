# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""mcast_transport -- per-zone attribution of `writer_mcast_send`.

Runs ONE (case, variant) with RMS_STAGE_ZONES=1 and MCZ=1 (the sub-zones inside
`writer_mcast_send`: `mcs_local` = the root's local publish + acked barrier,
`mcs_issue` = SenderPipe::send, i.e. the mcast + the flag + the source fence),
and prints:

  * per-zone n / total / mean / max, in ns;
  * the ROOT core's writer timeline, so the send's position on the critical path
    is visible rather than inferred;
  * every member's `writer_mcast_recv` END relative to the root's send START.

Env: MCT_CASE (default focus), MCT_VARIANT (default base_zoned).
"""

import os

os.environ["MCT_ZONES"] = "1"
os.environ.setdefault("MCT_CASES", os.environ.get("MCT_CASE", "focus"))
os.environ.setdefault("MCT_VARIANTS", os.environ.get("MCT_VARIANT", "base_zoned"))
os.environ.setdefault("MCT_REPS", "1")
os.environ.setdefault("MCT_TRIALS", "1")

import collections
import csv
import statistics
from pathlib import Path

CSVP = Path("generated/profiler/.logs/profile_log_device.csv")
if CSVP.exists():
    CSVP.unlink()

_here = Path(os.environ["MCT_DIR"])
exec(open(_here / "bench_mcast.py").read())

FREQ = 1.35  # cycles -> ns @ 1350 MHz

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
    for k in ("core_x", "core_y", "RISC processor type", "time[cycles since reset]", "zone name", "type", "run host ID")
}
target = sorted({int(r[ci["run host ID"]]) for r in rows})[-1]
stack = collections.defaultdict(list)
agg = collections.defaultdict(list)
events = []  # (core, risc, zone, start, end)
t0 = None
for r in rows:
    if int(r[ci["run host ID"]]) != target:
        continue
    key = (r[ci["core_x"]], r[ci["core_y"]], r[ci["RISC processor type"]])
    zone = r[ci["zone name"]].strip()
    typ = r[ci["type"]].strip()
    t = int(r[ci["time[cycles since reset]"]])
    t0 = t if t0 is None else min(t0, t)
    if typ == "ZONE_START":
        stack[(key, zone)].append(t)
    elif typ == "ZONE_END" and stack[(key, zone)]:
        st = stack[(key, zone)].pop()
        agg[(zone, key[2])].append(t - st)
        events.append((key[0] + "," + key[1], key[2], zone, st, t))

print(f"RESULT ---- zones (run {target}) ----")
print(f"RESULT {'zone':24s} {'risc':10s} {'n':>4s} {'tot_ns':>9s} {'mean_ns':>9s} {'max_ns':>9s}")
out = sorted(((sum(v), z, ri, len(v)) for (z, ri), v in agg.items()), reverse=True)
for tot, z, ri, n in out:
    v = agg[(z, ri)]
    print(f"RESULT {z:24s} {ri:10s} {n:4d} {tot/FREQ:9.0f} {statistics.mean(v)/FREQ:9.1f} {max(v)/FREQ:9.1f}")

# ---- the ROOT: the core that owns a writer_mcast_send ----
roots = sorted({e[0] for e in events if e[2] == "writer_mcast_send"})
print(f"RESULT ---- root core(s): {roots} ----")
for rt in roots[:1]:
    tl = sorted([e for e in events if e[0] == rt], key=lambda e: e[3])
    for c, ri, z, s, en in tl:
        print(f"RESULT root {ri:10s} {z:24s} {(s-t0)/FREQ:9.0f} -> {(en-t0)/FREQ:9.0f}   ({(en-s)/FREQ:7.0f} ns)")
    snd = [e for e in tl if e[2] == "writer_mcast_send"]
    if snd:
        s0 = snd[0][3]
        recv = sorted([(e[4] - s0) / FREQ for e in events if e[2] == "writer_mcast_recv"])
        if recv:
            print(
                f"RESULT recv_end_rel_to_send_start  n={len(recv)} min={recv[0]:.0f} "
                f"med={statistics.median(recv):.0f} max={recv[-1]:.0f} ns"
            )
        rs = sorted([(e[3] - s0) / FREQ for e in events if e[2] == "writer_mcast_recv"])
        print(
            f"RESULT recv_START_rel_to_send_start n={len(rs)} min={rs[0]:.0f} med={statistics.median(rs):.0f} max={rs[-1]:.0f} ns"
        )
# ---- kernel spans ----
spans = collections.defaultdict(lambda: [float("inf"), -float("inf")])
for c, ri, z, s, en in events:
    spans[(c, ri)][0] = min(spans[(c, ri)][0], s)
    spans[(c, ri)][1] = max(spans[(c, ri)][1], en)
byrisc = collections.defaultdict(list)
for (c, ri), (lo, hi) in spans.items():
    byrisc[ri].append((hi - t0) / FREQ)
print("RESULT ---- last zone END per RISC (ns from first zone start) ----")
for ri, v in sorted(byrisc.items()):
    print(f"RESULT end {ri:10s} n={len(v):3d} max={max(v):9.0f} med={statistics.median(v):9.0f}")
