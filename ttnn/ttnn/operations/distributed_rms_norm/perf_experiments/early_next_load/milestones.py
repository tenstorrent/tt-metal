"""Root milestones per (device, traced run) from a profile_log_device.csv.
usage: milestones.py <csv> [root x,y=1,9] [runs base ids=4096..9216]
Columns (us from the chip-run t0): pub0/pub1 = end of rd_wait_devpart (wave 0/1 own partial ready -> publish),
g0 = rd_gather(w0) start, g0e/g1e = rd_gather ends (all slots in), st0 = first wr_store start, end = latest BRISC-KERNEL
end over all cores, plus max over cores of the wave-1 x landed (end of 2nd rd_load / rd_load_fin)."""
import csv, sys, collections

path = sys.argv[1]
root = tuple(map(int, (sys.argv[2] if len(sys.argv) > 2 else "1,9").split(",")))
bases = list(range(4096, 9217, 1024))
with open(path) as fh:
    hdr = fh.readline()
    mhz = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rd = csv.reader(fh)
    next(rd)
    ev = collections.defaultdict(list)
    for r in rd:
        if len(r) > 12:
            ev[int(r[7])].append(r)
print("dev run   pub0  pub1   g0    g0e   g1e   st0   end   x1max")
for b in bases:
    for d in range(4):
        rows = ev.get(b + d, [])
        if not rows:
            continue
        t0 = min(int(r[5]) for r in rows if r[10].strip().endswith("-KERNEL") and r[11].strip() == "ZONE_START")
        open_ = {}
        z = collections.defaultdict(list)
        for r in sorted(rows, key=lambda r: int(r[5])):
            k = (int(r[1]), int(r[2]), r[3].strip(), r[10].strip())
            t = (int(r[5]) - t0) / mhz
            if r[11].strip() == "ZONE_START":
                open_.setdefault(k, []).append(t)
            elif open_.get(k):
                z[k].append((open_[k].pop(), t))
        g = lambda n, risc="NCRISC": z.get((root[0], root[1], risc, n), [])
        dp = g("rd_wait_devpart")
        ga = g("rd_gather")
        st = g("wr_store", "BRISC")
        end = max(e for k, v in z.items() if k[3] == "BRISC-KERNEL" for s, e in v)
        x1 = []
        for k, v in z.items():
            if k[2] == "NCRISC" and k[3] == "rd_load_fin":
                x1.append(v[0][1])
            elif (
                k[2] == "NCRISC"
                and k[3] == "rd_load"
                and len(v) >= 2
                and (k[0], k[1], "NCRISC", "rd_load_fin") not in z
            ):
                x1.append(v[1][1])
        f = lambda v, i, j: f"{v[i][j]:5.2f}" if len(v) > i else "  -  "
        print(
            f"{d}   {b+d:5d} {f(dp,0,1)} {f(dp,1,1)} {f(ga,0,0)} {f(ga,0,1)} {f(ga,1,1)} {f(st,0,0)} {end:5.2f} {max(x1) if x1 else 0:5.2f}"
        )
