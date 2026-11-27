"""Summarise a compute_drain bench profile: per program run (in run order = CD_VARIANTS order), the drain
(last writer block end - last chunk landed), kernel span, and per-row compute zones (us from kernel start).
Usage: summarize.py <csv> <variants comma list> [-v]"""
import csv, sys
from collections import defaultdict

F = 1350.0
path, names = sys.argv[1], [v for v in sys.argv[2].split(",") if v]
verbose = "-v" in sys.argv
rows = list(csv.reader(open(path)))[2:]
runs = defaultdict(list)
for r in rows:
    runs[int(r[7])].append(r)
ids = sorted(runs)
ids = ids[-len(names) :]
for name, rid in zip(names, ids):
    ev = defaultdict(list)
    kst, ken = [], []
    for r in runs[rid]:
        z, typ, risc, t = r[10].strip(), r[11].strip(), r[3].strip(), int(r[5])
        if z.endswith("-KERNEL"):
            (kst if typ == "ZONE_START" else ken).append(t)
        ev[(risc, z)].append((typ, t))
    t0 = min(kst)

    def spans(k):
        out, st = [], None
        for typ, t in sorted(ev.get(k, []), key=lambda a: a[1]):
            if typ == "ZONE_START":
                st = t
            elif st is not None:
                out.append(((st - t0) / F, (t - t0) / F))
                st = None
        return out

    landed = spans(("NCRISC", "cd_last_landed"))[0][0]
    wr = spans(("BRISC", "cd_wr_block"))
    end = wr[-1][1]
    print(
        f"{name:12s} span {(max(ken)-t0)/F:6.2f} us  last_landed {landed:6.2f}  last_write_end {end:6.2f}  DRAIN {end-landed:5.2f} us"
        f"  (last matmul pack end {spans(('TRISC_2','compute_matmul'))[-1][1]-landed:5.2f} after landing)"
    )
    if verbose:
        for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
            for z in ("compute_tilize", "compute_matmul"):
                s = spans((risc, z))
                print(f"   {risc} {z:15s} " + " ".join(f"{a:5.2f}-{b:5.2f}" for a, b in s))
        print("   BRISC wr_block      " + " ".join(f"{a:5.2f}-{b:5.2f}" for a, b in wr))
