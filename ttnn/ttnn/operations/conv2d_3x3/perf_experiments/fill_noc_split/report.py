# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# usage: report.py <profile_log_device.csv[.gz]> <ids file>
# Per program run (in order, matched to the test ids): kernel span (first kernel start to last kernel end = DEVICE
# KERNEL DURATION), the fill window (first fill_loop start to last fill_loop end over every core and RISC) and the
# per-RISC fill_loop duration p50 / max (us).
import collections, csv, gzip, statistics, sys

path, idfile = sys.argv[1], sys.argv[2]
ids = [l.strip() for l in open(idfile) if l.strip()]
f = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
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
n = 0
for run in sorted(runs):
    ev = runs[run]
    if not any(e[3] == "fill_loop" for e in ev):
        continue
    name = ids[n] if n < len(ids) else f"run{run}"
    n += 1
    ks = [e[2] for e in ev if e[3].endswith("-KERNEL") and e[4] == "ZONE_START"]
    ke = [e[2] for e in ev if e[3].endswith("-KERNEL") and e[4] == "ZONE_END"]
    st, dur = {}, collections.defaultdict(list)
    fs, fe = [], []
    own_st, own = {}, collections.defaultdict(float)
    for c, risc, t, z, k in sorted(ev, key=lambda e: e[2]):
        if z == "own":
            if k == "ZONE_START":
                own_st[(c, risc)] = t
            elif (c, risc) in own_st:
                own[(c, risc)] += (t - own_st.pop((c, risc))) / freq
            continue
        if z != "fill_loop":
            continue
        if k == "ZONE_START":
            st[(c, risc)] = t
            fs.append(t)
        elif (c, risc) in st:
            dur[risc].append((t - st.pop((c, risc))) / freq)
            fe.append(t)
    span = (max(ke) - min(ks)) / freq
    win = (max(fe) - min(fs)) / freq
    per = "  ".join(f"{risc[:2]} p50 {statistics.median(d):.0f} max {max(d):.0f}" for risc, d in sorted(dur.items()))
    owns = collections.defaultdict(list)
    for (c, risc), v in own.items():
        owns[risc].append(v)
    per_own = "  ".join(
        f"own {risc[:2]} p50 {statistics.median(d):.0f} max {max(d):.0f}" for risc, d in sorted(owns.items())
    )
    print(f"{name:26s} DKD {span * 1000:10.0f} ns  fill {win:8.1f} us  {per}  {per_own}")
