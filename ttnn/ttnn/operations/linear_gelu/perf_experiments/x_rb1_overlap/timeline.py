import csv, sys
from collections import defaultdict

p = sys.argv[1]
f = open(p)
first = f.readline()
freq = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
rows = list(csv.DictReader(f, skipinitialspace=True))
rids = sorted({int(r["run host ID"]) for r in rows})
rid = rids[int(sys.argv[2])]
rows = [r for r in rows if int(r["run host ID"]) == rid]
t0 = min(int(r["time[cycles since reset]"]) for r in rows)
ev = defaultdict(list)
op = {}
for r in rows:
    k = ((r["core_x"], r["core_y"]), r["RISC processor type"], r["zone name"])
    t = int(r["time[cycles since reset]"])
    if r["type"] == "ZONE_START":
        op[k] = t
    elif k in op:
        ev[k[0]].append((k[1], k[2], (op.pop(k) - t0) / freq, (t - t0) / freq))
end = {
    c: max(e[3] for e in v if e[1] == "TRISC-KERNEL") for c, v in ev.items() if any(e[1] == "TRISC-KERNEL" for e in v)
}
cores = [max(end, key=end.get)] + [tuple(a.split(",")) for a in sys.argv[3:]]
for c in cores:
    print("core", c, "end", round(end.get(c, 0), 1))
    for e in sorted(ev[c], key=lambda e: (e[0], e[2])):
        if "FW" in e[1]:
            continue
        print(f"  {e[0]:8s} {e[1]:22s} {e[2]:7.2f} -> {e[3]:7.2f} ({e[3]-e[2]:6.2f})")
