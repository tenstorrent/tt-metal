"""Per-RISC kernel/firmware start/end offsets (ns, relative to the earliest kernel start) of each launch of a
1-core program, median over launches 2..N. usage: timeline.py profile_log_device.csv"""
import csv, sys, statistics
from collections import defaultdict

f = sys.argv[1]
rows = list(csv.reader(open(f)))[2:]
launch = defaultdict(lambda: defaultdict(dict))  # run_id -> (core,risc,zone) -> {start,end}
for r in rows:
    r = [x.strip() for x in r]
    core = (r[1], r[2])
    risc = r[3]
    t = int(r[5])
    run = r[7]
    zone = r[10]
    typ = r[11]
    k = (core, risc, zone)
    d = launch[run][k]
    d["s" if typ == "ZONE_START" else "e"] = t
runs = sorted(launch, key=int)[1:]
acc = defaultdict(list)
for run in runs:
    z = launch[run]
    t0 = min(v["s"] for k, v in z.items() if k[2].endswith("KERNEL"))
    for k, v in z.items():
        if "s" in v and "e" in v:
            acc[(k[1], k[2])].append(((v["s"] - t0) / 1.35, (v["e"] - t0) / 1.35))
order = ["BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"]
for (risc, zone), v in sorted(
    acc.items(), key=lambda kv: (kv[0][1], order.index(kv[0][0]) if kv[0][0] in order else 9)
):
    s = statistics.median(x[0] for x in v)
    e = statistics.median(x[1] for x in v)
    print(f"{zone:14s} {risc:8s} start={s:7.0f} end={e:7.0f} dur={e - s:6.0f}  (n={len(v)})")
