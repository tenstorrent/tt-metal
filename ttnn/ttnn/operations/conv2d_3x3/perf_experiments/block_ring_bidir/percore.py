# Per-core zone sums (us) for one run of a device profile log: grid of core (x, y) -> value.
import collections, csv, sys

path, run_sel, risc_sel, zone_sel = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
with open(path) as f:
    header = f.readline()
    freq = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = csv.reader(f)
    cols = [c.strip() for c in next(rows)]
    ix = {c: i for i, c in enumerate(cols)}
    st = collections.defaultdict(list)
    tot = collections.defaultdict(float)
    first = {}
    for r in rows:
        if len(r) < len(cols) or int(r[ix["run host ID"]]) != run_sel:
            continue
        risc = r[ix["RISC processor type"]].strip()
        zone = r[ix["zone name"]].strip()
        if risc != risc_sel or zone != zone_sel:
            continue
        core = (int(r[ix["core_x"]]), int(r[ix["core_y"]]))
        t = int(r[ix["time[cycles since reset]"]])
        kind = r[ix["type"]].strip()
        if kind == "ZONE_START":
            st[core].append(t)
        elif kind == "ZONE_END" and st[core]:
            tot[core] += (t - st[core].pop()) / freq
xs = sorted({c[0] for c in tot})
ys = sorted({c[1] for c in tot})
print("y\\x " + " ".join(f"{x:>6}" for x in xs))
for y in ys:
    print(f"{y:>3} " + " ".join(f"{tot.get((x, y), float('nan')):6.2f}" for x in xs))
