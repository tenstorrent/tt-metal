# usage: percore.py <profile_log_device.csv> <run_index (1-based among runs with >= 8 cores)> [zone,zone,...]
# Per-core zone end/start times (us from the run's earliest kernel start), sorted by TRISC kernel end; plus quantiles.
import collections, csv, sys

path, idx = sys.argv[1], int(sys.argv[2])
with open(path) as f:
    hdr = f.readline()
    freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = csv.reader(f)
    cols = [c.strip() for c in next(rows)]
    ix = {c: i for i, c in enumerate(cols)}
    runs = collections.defaultdict(list)
    for r in rows:
        if len(r) < len(cols):
            continue
        runs[int(r[ix["run host ID"]])].append(
            (
                (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                r[ix["RISC processor type"]].strip(),
                int(r[ix["time[cycles since reset]"]]),
                r[ix["zone name"]].strip(),
                r[ix["type"]].strip(),
            )
        )
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
ev = sorted(runs[big[idx - 1]], key=lambda e: e[2])
t0 = min(t for c, r, t, z, k in ev if z.endswith("-KERNEL") and k == "ZONE_START")
us = lambda c: (c - t0) / freq
span = collections.defaultdict(list)  # (core, risc, zone) -> [(s, e)]
stack = collections.defaultdict(list)
for core, risc, t, z, kind in ev:
    if kind == "ZONE_START":
        stack[(core, risc, z)].append(t)
    elif stack[(core, risc, z)]:
        s = stack[(core, risc, z)].pop()
        span[(core, risc, z)].append((us(s), us(t)))
cores = sorted({k[0] for k in span})
want = sys.argv[3].split(",") if len(sys.argv) > 3 else None
keys = sorted({(k[1], k[2]) for k in span if not k[2].endswith("-FW")})
if want:
    keys = [k for k in keys if k[1] in want or k[1].endswith("-KERNEL")]


def end_of(c, k):
    v = span.get((c, k[0], k[1]))
    return (v[0][0], v[-1][1]) if v else None


tk = ("TRISC_0", "TRISC-KERNEL")
order = sorted(cores, key=lambda c: end_of(c, tk)[1] if end_of(c, tk) else 0)
print(f"run {big[idx-1]} cores {len(cores)}")
short = (
    lambda k: k[0][:2]
    + ":"
    + k[1].replace("compute_", "c.").replace("reader_", "r.").replace("writer_", "w.").replace("-KERNEL", "K")
)
print("core      " + " ".join(f"{short(k):>22}" for k in keys))
sel = order[:3] + order[len(order) // 2 - 1 : len(order) // 2 + 2] + order[-8:]
for c in sel:
    cells = []
    for k in keys:
        v = end_of(c, k)
        cells.append(f"{v[0]:5.2f}-{v[1]:5.2f}".rjust(22) if v else " " * 22)
    print(f"{str(c):9} " + " ".join(cells))
print("quantiles of end (p10 p50 p90 max):")
for k in keys:
    ends = sorted(end_of(c, k)[1] for c in cores if end_of(c, k))
    starts = sorted(end_of(c, k)[0] for c in cores if end_of(c, k))
    dur = sorted(sum(e - s for s, e in span[(c, k[0], k[1])]) for c in cores if end_of(c, k))
    q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]
    print(
        f"  {short(k):>26} end {q(ends,.1):5.2f} {q(ends,.5):5.2f} {q(ends,.9):5.2f} {ends[-1]:5.2f}  start p50 {q(starts,.5):5.2f}  dur p50 {q(dur,.5):5.2f} max {dur[-1]:5.2f}"
    )
