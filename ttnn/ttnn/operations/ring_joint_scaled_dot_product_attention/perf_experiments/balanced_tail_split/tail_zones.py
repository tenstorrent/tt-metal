# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-zone medians of the tail piece's sampled blocks vs the whole units' (chip 0, last run)."""
import pandas as pd

LOG = "generated/profiler/.logs/profile_log_device.csv"
df = pd.read_csv(LOG, skiprows=1, skipinitialspace=True)
df.columns = [c.strip() for c in df.columns]
run_col = "run host ID" if "run host ID" in df.columns else "run ID"
df = df[df[run_col] == df.groupby("PCIe slot")[run_col].transform("max")]
d = df[df["PCIe slot"] == df["PCIe slot"].min()]
t = "time[cycles since reset]"
z = d[d["type"].isin(["ZONE_START", "ZONE_END"]) & ~d["zone name"].astype(str).str.contains("KERNEL|FW")]
rows = []
for (zn, risc, cx, cy), g in z.groupby(["zone name", "RISC processor type", "core_x", "core_y"]):
    g = g.sort_values(t)
    st = g[g["type"] == "ZONE_START"][t].values
    en = g[g["type"] == "ZONE_END"][t].values
    n = min(len(st), len(en))
    for k in range(n):
        rows.append((zn, risc, cx, cy, k, (en[k] - st[k]) / 1.35, st[k]))
r = pd.DataFrame(rows, columns=["zone", "risc", "x", "y", "k", "ns", "start"])
cu = r[r.zone == "compute_unit"].groupby(["x", "y"]).size()
tail_cores = set(cu[cu == cu.max()].index)
r["tail_core"] = [(a, b) in tail_cores for a, b in zip(r.x, r.y)]
# sampled per-block zones: first 6 (or 5 for kb>0 zones) executions = unit 0; later ones = tail piece
r["second_window"] = (
    r.groupby(["zone", "risc", "x", "y"])["start"].rank()
    > r.groupby(["zone", "risc", "x", "y"])["start"].transform("count") / 2
)
r2 = r[r.tail_core]
for (zn, risc), g in r2.groupby(["zone", "risc"]):
    if risc not in ("TRISC_1", "NCRISC"):
        continue
    a = g[~g.second_window].ns.median()
    b = g[g.second_window].ns.median()
    print(f"{zn:28s} {risc:8s} first-half med {a:12.0f}  second-half (tail) med {b:12.0f}")
