# usage: coremap.py <dev csv[.gz]> <ids file> <test id> [zone=fill_loop]: per-core fill_loop duration (us) on the
# physical grid, NCRISC and BRISC, for one run of the session (a zone that repeats is summed per core).
import collections, csv, gzip, sys

path, idfile, want = sys.argv[1], sys.argv[2], sys.argv[3]
zone = sys.argv[4] if len(sys.argv) > 4 else "fill_loop"
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
    runs[int(r[ix["run host ID"]])].append(r)
fr = [k for k in sorted(runs) if any(r[ix["zone name"]].strip() == zone for r in runs[k])]
run = fr[ids.index(want)]
st, d = {}, {}
for r in sorted(runs[run], key=lambda r: int(r[ix["time[cycles since reset]"]])):
    if r[ix["zone name"]].strip() != zone:
        continue
    key = (int(r[ix["core_x"]]), int(r[ix["core_y"]]), r[ix["RISC processor type"]].strip())
    t = int(r[ix["time[cycles since reset]"]])
    if r[ix["type"]].strip() == "ZONE_START":
        st[key] = t
    else:
        d[key] = d.get(key, 0.0) + (t - st[key]) / freq / 1000
xs = sorted({k[0] for k in d})
ys = sorted({k[1] for k in d})
for risc in sorted({k[2] for k in d}):
    print(f"{want} {risc} {zone} ms; columns x = {xs}")
    for y in ys:
        print(f"y={y:2d} " + " ".join(f"{d[(x, y, risc)]:5.2f}" if (x, y, risc) in d else "    -" for x in xs))
