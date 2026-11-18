# Perf 1 tail chain per call (calls after the first): last bulk packet handed to the router -> last landing
# notification -> last normalize (pack) -> last store, per chip, us from the chip's earliest kernel start.
import csv, sys, collections, statistics as st

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
F = 1350.0
prog = collections.defaultdict(list)
with open(path) as f:
    next(f)
    r = csv.reader(f)
    next(r)
    for x in r:
        prog[(x[0], int(x[7]))].append((int(x[1]), int(x[2]), x[3], int(x[5]), x[10], x[11]))
out = collections.defaultdict(list)
for (chip, run), ev in sorted(prog.items()):
    if "sender_fence" not in {e[4] for e in ev}:
        continue
    t0 = min(e[3] for e in ev if e[4].endswith("-KERNEL") and e[5] == "ZONE_START")
    end = lambda n, risc=None: max(
        (e[3] for e in ev if e[4] == n and e[5] == "ZONE_END" and (risc is None or e[2] == risc)), default=t0
    )
    kend = max(e[3] for e in ev if e[4].endswith("-KERNEL") and e[5] == "ZONE_END")
    out[chip].append(
        [
            (v - t0) / F
            for v in (
                end("sender_fence"),
                end("sender_bulk"),
                end("recv_notify"),
                end("compute_normalize", "TRISC_2"),
                end("writer_store"),
                kend,
            )
        ]
    )
print("chip  fence_end  bulk_end  last_land  last_norm  last_store  kernel_end   (medians over calls 2..)")
for chip in sorted(out):
    rows = out[chip][1:]
    med = [st.median(c) for c in zip(*rows)]
    print(chip, "  ".join(f"{m:8.2f}" for m in med))
