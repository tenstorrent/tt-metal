# usage: acks.py <profile_log_device.csv> <case index (1-based among runs with >= 8 cores)>
# Per-core first writer_weight_fetch start (just after the receiver's start ack) and kernel starts, in us from the
# run's earliest kernel start; plus per-receiver round-arrival (weight_fetch ends) percentiles.
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
t0 = min(e[2] for e in ev if e[3].endswith("-KERNEL") and e[4] == "ZONE_START")
us = lambda c: (c - t0) / freq
kst = {}
fetch = collections.defaultdict(list)
wait = collections.defaultdict(list)
st = {}
for core, risc, t, z, kind in ev:
    if z == "BRISC-KERNEL" and kind == "ZONE_START":
        kst[core] = us(t)
    if z in ("writer_weight_fetch", "compute_kwf_weight_wait") or z.startswith("server_"):
        key = (core, risc, z)
        if kind == "ZONE_START":
            st[key] = us(t)
        elif key in st:
            (fetch if z == "writer_weight_fetch" else wait)[(core, z)].append((st.pop(key), us(t)))
recv = sorted(c for c, z in fetch if z == "writer_weight_fetch")


def pct(v):
    v = sorted(v)
    return f"min {v[0]:.2f} p50 {v[len(v)//2]:.2f} max {v[-1]:.2f}"


print("receivers", len(recv))
print("BRISC kernel start", pct([kst[c] for c in recv]))
print("first fetch start (after start ack)", pct([fetch[(c, "writer_weight_fetch")][0][0] for c in recv]))
n = min(len(fetch[(c, "writer_weight_fetch")]) for c in recv)
for r in range(n):
    print(f"round {r} lands (fetch end)", pct([fetch[(c, "writer_weight_fetch")][r][1] for c in recv]))
