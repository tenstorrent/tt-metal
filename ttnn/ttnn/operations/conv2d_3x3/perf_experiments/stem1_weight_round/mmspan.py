# usage: mmspan.py <profile_log_device.csv> <run_index> : per-core matmul start (end of the last compute wait zone on
# TRISC_0: compute_slab_wait / compute_sd_weight_wait) and end (compute_matmul_stream end), quantiles; plus the
# critical core (latest TRISC_0 end).
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
stack = collections.defaultdict(list)
spans = collections.defaultdict(list)
for core, risc, t, z, kind in ev:
    if risc != "TRISC_0":
        continue
    if kind == "ZONE_START":
        stack[(core, z)].append(t)
    elif stack[(core, z)]:
        s = stack[(core, z)].pop()
        spans[(core, z)].append((us(s), us(t)))
cores = sorted({c for c, z in spans})
res = []
for c in cores:
    waits = [e for z in ("compute_slab_wait", "compute_sd_weight_wait", "") for s, e in spans.get((c, z), [])]
    mm = spans.get((c, "compute_matmul_stream"))
    if not mm or not waits:
        continue
    st = max(waits)
    res.append((mm[0][1], st, c))
res.sort()
q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]
starts = sorted(r[1] for r in res)
ends = sorted(r[0] for r in res)
dur = sorted(r[0] - r[1] for r in res)
print(
    f"run {big[idx-1]}: mm start p10/p50/p90/max {q(starts,.1):.2f} {q(starts,.5):.2f} {q(starts,.9):.2f} {starts[-1]:.2f}"
    f" | end {q(ends,.1):.2f} {q(ends,.5):.2f} {q(ends,.9):.2f} {ends[-1]:.2f}"
    f" | end - last wait p10/p50/max {q(dur,.1):.2f} {q(dur,.5):.2f} {dur[-1]:.2f}"
)
e, s, c = res[-1]
print(f"  critical core {c}: last wait end {s:.2f}, matmul end {e:.2f}")
