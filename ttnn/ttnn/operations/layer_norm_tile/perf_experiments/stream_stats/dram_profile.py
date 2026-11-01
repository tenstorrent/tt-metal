import csv, sys
from collections import defaultdict

path, want = sys.argv[1], sys.argv[2]
with open(path) as f:
    header = f.readline()
    freq = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
open_ = {}
ev = defaultdict(list)
for r in rows[1:]:
    r = [c.strip() for c in r]
    if r[idx["run host ID"]] != want:
        continue
    core = (int(r[idx["core_x"]]), int(r[idx["core_y"]]))
    risc = r[idx["RISC processor type"]]
    z = r[idx["zone name"]]
    typ = r[idx["type"]]
    t = int(r[idx["time[cycles since reset]"]])
    k = (core, risc, z)
    if typ == "ZONE_START":
        open_[k] = t
    elif typ == "ZONE_END" and k in open_:
        ev[core].append((risc, z, open_.pop(k), t))
t0 = min(s for c in ev for (_, z, s, _) in ev[c] if z.endswith("KERNEL"))
us = lambda t: (t - t0) / freq
B = 48 * 1024
bins_r = defaultdict(float)
bins_w = defaultdict(float)


def spread(bins, a, b, nbytes):
    if b <= a:
        b = a + 0.01
    t = a
    while t < b:
        nxt = min(int(t) + 1, b)
        bins[int(t)] += nbytes * (nxt - t) / (b - a)
        t = nxt


for c, e in ev.items():
    for risc, z, s, t in e:
        if z == "writer_residual_block":
            spread(bins_r, us(s), us(t), B)
        if z == "writer_store_block":
            # writer waits for output; approximate active write from the first shift start after s
            shifts = [
                us(x[2]) for x in e if x[1] == "compute_affine_shift" and x[0] == "TRISC_1" and us(x[3]) >= us(s) - 0.5
            ]
            a = max(us(s), min(shifts) if shifts else us(s))
            spread(bins_w, a, us(t), B)
mx = int(max(list(bins_r) + list(bins_w))) + 1
for i in range(mx):
    r, w = bins_r[i] / 1e3, bins_w[i] / 1e3  # KB per us == MB/s*1e-3 -> GB/s
    print(f"{i:3d} us  read {r:6.0f} GB/s  write {w:6.0f} GB/s  total {r+w:6.0f}  " + "#" * int((r + w) / 20))
