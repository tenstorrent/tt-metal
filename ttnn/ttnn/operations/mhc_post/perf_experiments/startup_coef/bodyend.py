"""bodyend.py <log> <calls> : per run, last compute_mix end (TRISC_1) per core: median, max and the 6 latest
cores; and the per-core sum of reader_barrier for those cores."""
import collections, csv, json, statistics, sys

fh = open(sys.argv[1])
freq = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
rows = list(csv.DictReader(fh, skipinitialspace=True))
runs = collections.defaultdict(list)
for r in rows:
    runs[r["run host ID"]].append(r)
ours = [k for k in sorted(runs, key=int) if any(r["zone name"] == "reader_coef_share" for r in runs[k])]
labs = [json.loads(l) for l in open(sys.argv[2])]
for i, k in enumerate(ours):
    t0 = min(int(r["time[cycles since reset]"]) for r in runs[k] if r["zone name"].endswith("-KERNEL"))
    mixend = {}
    starts = {}
    barr = collections.defaultdict(float)
    for r in runs[k]:
        t = int(r["time[cycles since reset]"])
        c = (int(r["core_x"]), int(r["core_y"]))
        key = (c, r["RISC processor type"], r["zone name"])
        if r["type"] == "ZONE_START":
            starts[key] = t
        elif key in starts:
            if r["zone name"] == "compute_mix" and r["RISC processor type"] == "TRISC_1":
                mixend[c] = (t - t0) / freq
            if r["zone name"] == "reader_barrier":
                barr[c] += (t - starts[key]) / freq
    v = sorted(mixend.values())
    top = sorted(mixend.items(), key=lambda x: -x[1])[:6]
    print(
        f"{labs[i]['case'][:20]} {labs[i]['who']}: mix end med {statistics.median(v):5.1f} max {v[-1]:5.1f} | "
        + " ".join(f"{c}:{e:.1f}/b{barr[c]:.1f}" for c, e in top)
    )
