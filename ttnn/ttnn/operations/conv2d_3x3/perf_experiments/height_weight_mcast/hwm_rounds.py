# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-round receive timing over all cores of one dispatch (no device, no ttnn import).

    python3 hwm_rounds.py <profile_log_device.csv> <dispatch index among runs with >= 8 cores> [zone]

For the zone (default writer_weight_fetch on BRISC) prints, per execution index i, the start/end (us from the
dispatch's first marker) min / p50 / max over cores, and the cores holding the max end."""
import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import zone_report  # noqa: E402

path, idx = sys.argv[1], int(sys.argv[2])
zone = sys.argv[3] if len(sys.argv) > 3 else "writer_weight_fetch"
freq, runs = zone_report.load(path)
run_ids = sorted(rid for rid, ev in runs.items() if len({c for c, *_ in ev}) >= 8)
ev = runs[run_ids[idx]]
t0 = min(e[2] for e in ev)
per_core = collections.defaultdict(list)
stack = collections.defaultdict(list)
for c, risc, t, z, kind in sorted(ev, key=lambda e: e[2]):
    if z != zone:
        continue
    if kind == "ZONE_START":
        stack[c].append(t)
    elif kind == "ZONE_END" and stack[c]:
        s = stack[c].pop()
        per_core[c].append(((s - t0) / freq, (t - t0) / freq))
n = max(len(v) for v in per_core.values())
for i in range(n):
    rows = [(c, v[i]) for c, v in per_core.items() if len(v) > i]
    starts = [r[1][0] for r in rows]
    ends = [r[1][1] for r in rows]
    late = sorted(rows, key=lambda r: -r[1][1])[:3]
    print(
        f"{zone}[{i}] cores {len(rows)}  start min {min(starts):6.2f} p50 {statistics.median(starts):6.2f} max {max(starts):6.2f}"
        f"  end min {min(ends):6.2f} p50 {statistics.median(ends):6.2f} max {max(ends):6.2f}  late {[(c, round(v[1], 2)) for c, v in late]}"
    )
