# usage: zstat.py <profile_log_device.csv[.gz]> <run_index> <zone> <risc> : per-core sum of a zone's durations and the
# per-core k-th occurrence end times (us from the run's earliest kernel start), as quantiles over cores.
import collections, csv, gzip, sys

path, idx, zname, risc = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
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
t0 = min(t for c, r, t, z, k in ev if z.endswith("-KERNEL") and k == "ZONE_START")
st = {}
per = collections.defaultdict(list)
for core, r, t, z, k in ev:
    if z != zname or r != risc:
        continue
    if k == "ZONE_START":
        st[core] = t
    elif core in st:
        per[core].append(((st.pop(core) - t0) / freq, (t - t0) / freq))
sums = sorted(sum(e - s for s, e in v) for v in per.values())
q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]
print(f"{zname} {risc}: cores {len(per)} sum p50 {q(sums, .5):.2f} p90 {q(sums, .9):.2f} max {sums[-1]:.2f}")
k = max(len(v) for v in per.values())
for i in range(k):
    ends = sorted(v[i][1] for v in per.values() if len(v) > i)
    print(f"  occurrence {i}: end p10 {q(ends, .1):.2f} p50 {q(ends, .5):.2f} p90 {q(ends, .9):.2f} max {ends[-1]:.2f}")
