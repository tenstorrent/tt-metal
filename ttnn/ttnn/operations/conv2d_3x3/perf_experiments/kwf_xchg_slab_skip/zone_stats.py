# usage: zone_stats.py <profile_log_device.csv> <run_index> <zone> [risc]: per-core first start / last end of a zone
# (us from the run's earliest kernel start): min / median / max over cores, and the worst cores.
import collections, csv, statistics, sys

path, idx, zone = sys.argv[1], int(sys.argv[2]), sys.argv[3]
risc_want = sys.argv[4] if len(sys.argv) > 4 else None
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
ev = runs[big[idx - 1]]
t0 = min(e[2] for e in ev if e[3].endswith("-KERNEL") and e[4] == "ZONE_START")
st, en, n = {}, {}, collections.Counter()
for core, risc, t, z, kind in ev:
    if z != zone or (risc_want and risc != risc_want):
        continue
    if kind == "ZONE_START":
        st[core] = min(st.get(core, t), t)
        n[core] += 1
    else:
        en[core] = max(en.get(core, t), t)
us = lambda c: (c - t0) / freq
s = sorted(us(v) for v in st.values())
e = sorted(us(v) for v in en.values())
if not e:
    print(zone, "none")
    sys.exit()
print(
    f"{zone}: cores {len(e)} count/core {statistics.median(n.values())}  start min/med/max {s[0]:.2f}/{statistics.median(s):.2f}/{s[-1]:.2f}  end min/med/max {e[0]:.2f}/{statistics.median(e):.2f}/{e[-1]:.2f}"
)
