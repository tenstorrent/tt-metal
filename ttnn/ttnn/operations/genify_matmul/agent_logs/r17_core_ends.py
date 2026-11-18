"""Per-core kernel-zone end time (us from the op's earliest start) for one RISC, laid out on the grid.
usage: r17_core_ends.py <profile_log_device.csv> <risc e.g. NCRISC> [zone, default <risc>-KERNEL]"""
import csv, sys, collections

rows = list(csv.reader(open(sys.argv[1])))[2:]
risc = sys.argv[2]
zone = sys.argv[3] if len(sys.argv) > 3 else f"{risc}-KERNEL"
by_run = collections.defaultdict(list)
for r in rows:
    by_run[r[7].strip()].append(r)
run = max(by_run, key=lambda k: len(by_run[k]))
t0 = min(int(r[5]) for r in by_run[run])
end = {}
for r in by_run[run]:
    if r[3].strip() == risc and r[10].strip() == zone and r[11].strip() == "ZONE_END":
        end[(int(r[1]), int(r[2]))] = (int(r[5]) - t0) / 1350
xs = sorted({c[0] for c in end})
ys = sorted({c[1] for c in end})
print("phys y\\x " + " ".join(f"{x:5d}" for x in xs))
for y in ys:
    print(f"{y:8d} " + " ".join(f"{end[(x, y)]:5.1f}" if (x, y) in end else "    -" for x in xs))
