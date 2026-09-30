"""Per-core head/tail timeline from profile_log_device.csv (one run host ID): kernel start -> first compute_mix
start (head), last reader barrier end -> kernel end (tail), for the slowest / median / fastest cores."""
import csv, sys, statistics
from collections import defaultdict

path, run = sys.argv[1], sys.argv[2]
MHZ = 1350.0
ev = defaultdict(list)
with open(path) as f:
    next(f)
    r = csv.reader(f)
    next(r)
    for row in r:
        row = [x.strip() for x in row]
        if row[7] != run:
            continue
        ev[(row[1], row[2])].append((int(row[5]), row[3], row[10], row[11]))
rows = []
for core, es in ev.items():
    es.sort()
    k0 = min(t for t, risc, z, ty in es if z.endswith("-KERNEL") and ty == "ZONE_START")
    kend = max(t for t, risc, z, ty in es if z.endswith("-KERNEL") and ty == "ZONE_END")
    mix0 = min(
        (t for t, risc, z, ty in es if z == "compute_mix" and risc == "TRISC_1" and ty == "ZONE_START"), default=k0
    )
    mixend = max(
        (t for t, risc, z, ty in es if z == "compute_mix" and risc == "TRISC_1" and ty == "ZONE_END"), default=kend
    )
    rd_end = max((t for t, risc, z, ty in es if z == "reader_barrier" and ty == "ZONE_END"), default=k0)
    rows.append(((kend - k0) / MHZ, (mix0 - k0) / MHZ, (rd_end - k0) / MHZ, (mixend - k0) / MHZ, core))
rows.sort()
print("   wall   head(mix0)  reads_done  mix_done   core")
for w in rows[-5:] + [rows[len(rows) // 2]] + rows[:3]:
    print("  %6.1f  %8.1f  %9.1f  %9.1f   %s" % w)
print(
    "median head %.1f  median (wall - reads_done) %.1f"
    % (statistics.median(r[1] for r in rows), statistics.median(r[0] - r[2] for r in rows))
)
