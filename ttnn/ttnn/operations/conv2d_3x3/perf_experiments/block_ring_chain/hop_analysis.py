# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ring-hop decomposition from a RING_CHAIN_PROBE profile (host only).

    python3 hop_analysis.py <profile_log_device.csv> <run_index> <ring: x0,y0 x1,y1 ...> [--steps 8 --groups 2]

Per core (own clock): pack end -> reader sees the push (wait_step end), -> forward issued (r_rc_issue end), -> arrival
sent (r_rc_ret start = the write's ack seen). Across a hop (two clocks): arrival sent on c -> c+1's packer released
(c_rc_tok end) for the same partial; the ring sum of these over the cycle cancels the per-core clock offsets.
"""
import collections
import csv
import statistics
import sys

path, idx = sys.argv[1], int(sys.argv[2])
ring = [tuple(int(v) for v in a.split(",")) for a in sys.argv[3:] if "," in a]
with open(path) as f:
    hdr = f.readline()
    freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = csv.reader(f)
    cols = [c.strip() for c in next(rows)]
    ix = {c: i for i, c in enumerate(cols)}
    runs = collections.defaultdict(list)
    for r in rows:
        if len(r) < len(cols):
            continue
        runs[int(r[ix["run host ID"]])].append(
            (
                (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                r[ix["RISC processor type"]].strip(),
                int(r[ix["time[cycles since reset]"]]),
                r[ix["zone name"]].strip(),
                r[ix["type"]].strip(),
            )
        )
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
ev = sorted(runs[big[idx - 1]], key=lambda e: e[2])
t0 = min(e[2] for e in ev)
us = lambda c: (c - t0) / freq
stack = collections.defaultdict(list)
spans = collections.defaultdict(list)
for core, risc, t, z, kind in ev:
    if core not in ring:
        continue
    if kind == "ZONE_START":
        stack[(core, risc, z)].append(t)
    elif stack[(core, risc, z)]:
        s = stack[(core, risc, z)].pop()
        spans[(core, risc, z)].append((us(s), us(t)))


def sp(core, risc, z):
    return sorted(spans.get((core, risc, z), []))


k = len(ring)
per = collections.defaultdict(list)
hop = []
for ci, core in enumerate(ring):
    pack = sp(core, "TRISC_2", "c_rc_pack")
    tok = sp(core, "TRISC_2", "c_rc_tok")
    mm = sp(core, "TRISC_1", "c_rc_mm")
    seen = sp(core, "NCRISC", "reader_ring_wait_step")
    issue = sp(core, "NCRISC", "r_rc_issue")
    ret = sp(core, "NCRISC", "r_rc_ret")
    n = len(tok)
    for i in range(min(len(pack), len(seen))):
        per["pack_end->reader_sees"].append(seen[i][1] - pack[i][1])
        per["pack"].append(pack[i][1] - pack[i][0])
        per["tok_wait"].append(tok[i][1] - tok[i][0])
    for i in range(min(len(issue), len(ret))):
        per["issue_end->ack(ret)"].append(ret[i][0] - issue[i][1])
        per["reader_sees->issue_end"].append(issue[i][1] - seen[i][1])
    # hop: ret[i] on core (event i = step j group g, j < k-1) -> next core's tok[i + groups] (step j+1, same g)
    nxt = ring[(ci + 1) % k]
    ntok = sp(nxt, "TRISC_2", "c_rc_tok")
    groups = n // k if n else 0
    for i in range(len(ret)):
        if i + groups < len(ntok):
            hop.append((ci, i, ntok[i + groups][1] - ret[i][0], ntok[i + groups][1] - ntok[i + groups][0]))
for name, v in per.items():
    print(f"{name:28s} n={len(v):3d} p50 {statistics.median(v):6.3f} us  min {min(v):6.3f}  max {max(v):6.3f}")
# ring-average hop (offsets cancel over a full cycle, same event index on every core)
byi = collections.defaultdict(list)
for ci, i, d, w in hop:
    byi[i].append((d, w))
print(
    "ack(ret) on c -> packer of c+1 released, ring mean per event (offsets cancel); and how many of the k waited (>0.05us):"
)
for i in sorted(byi):
    v = byi[i]
    if len(v) == k:
        print(
            f"  event {i:2d}: mean {statistics.mean(d for d, _ in v):6.3f} us   waited {sum(1 for _, w in v if w > 0.05)}/{k}"
        )
