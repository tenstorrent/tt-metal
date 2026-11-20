# usage: trace.py <csv> <run> [cores x,y;x,y]: per core, the BFB_TRACE timestamps (us after the run's earliest kernel
# start): reader batch issue / land, writer batch issue / land, plus the tap build's kernel-row ends.
import collections, csv, sys

path, idx = sys.argv[1], int(sys.argv[2])
want = [tuple(int(v) for v in c.split(",")) for c in sys.argv[3].split(";")] if len(sys.argv) > 3 else None
with open(path) as f:
    hdr = f.readline()
    freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rd = csv.reader(f)
    cols = [c.strip() for c in next(rd)]
    ix = {c: i for i, c in enumerate(cols)}
    runs = collections.defaultdict(list)
    for r in rd:
        if len(r) < len(cols):
            continue
        runs[int(r[ix["run host ID"]])].append(
            (
                (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                r[ix["RISC processor type"]].strip(),
                int(r[ix["time[cycles since reset]"]]),
                r[ix["zone name"]].strip(),
                r[ix["type"]].strip(),
                r[ix["data"]].strip(),
            )
        )
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
ev = sorted(runs[big[idx - 1]], key=lambda e: e[2])
t0 = min(e[2] for e in ev if e[3].endswith("-KERNEL") and e[4] == "ZONE_START")
us = lambda v: (v - t0) / freq
per = collections.defaultdict(lambda: collections.defaultdict(dict))
kh = collections.defaultdict(list)
for c, r, t, z, k, d in ev:
    if z.startswith("bfb_"):
        per[c][z][int(d)] = us(t)
    if z == "compute_ts_kh" and r == "TRISC_0" and k == "ZONE_END":
        kh[c].append(us(t))
cores = want or sorted(per)
for c in cores:
    p = per[c]

    def row(name):
        dd = p.get(name, {})
        return " ".join(f"{dd[i]:4.1f}" for i in sorted(dd))

    print(f"{c}: kh ends {' '.join(f'{v:4.1f}' for v in kh[c])}")
    print(f"   r issue {row('bfb_rissue')} | land {row('bfb_rland')}")
    print(f"   w issue {row('bfb_wissue')} | land {row('bfb_wland')}")
