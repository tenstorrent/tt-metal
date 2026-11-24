# Per-port send / port-read spans (last profiled call per device) from a zones CSV.
import csv, collections, sys

for f in sys.argv[1:]:
    rows = list(csv.reader(open(f)))[2:]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r[0], r[7])].append((int(r[1]), int(r[2]), r[3], r[10], r[11], int(r[5])))
    print("==", f)
    for dev in sorted({k[0] for k in by}):
        last = max((k[1] for k in by if k[0] == dev), key=int)
        ev = by[(dev, last)]
        t0 = min(t for *_, n, ty, t in ev if n.endswith("-KERNEL") and ty == "ZONE_START")
        st = {}
        out = []
        for x, y, risc, n, ty, t in ev:
            if n not in ("psd_send", "prd_all"):
                continue
            k = (x, y, n)
            if ty == "ZONE_START":
                st[k] = t
            elif k in st:
                out.append((n, x, y, (st[k] - t0) / 1350, (t - t0) / 1350))
        print(" dev", dev, "  ".join(f"{n}@({x},{y}) {s:.1f}-{e:.1f}" for n, x, y, s, e in sorted(out)))
