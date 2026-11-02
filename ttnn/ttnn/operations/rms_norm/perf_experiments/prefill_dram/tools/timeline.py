"""Aggregate DRAM read / write rate over time for one GenericOp run (TILE interleaved k = 1 prefill).

usage: timeline.py <ops.csv> <profile_log_device.csv> <generic-op index> <bytes per tile-row> [bin_us]
Reads: each Tensix core's NCRISC rd_read zones (one per block); bytes spread uniformly over the zone.
Writes: each core's TRISC_2 cmp_pass2 zones (pass 2 is throttled by the writer's 2-unit cb_out).
Rows per core = rd_read zone count x (block rows) is not in the log, so bytes per zone = tile-row bytes x rows
derived from the zone length ratio is NOT attempted: pass the per-core rows via the reader RT args? -> we use
the op's blocking: rows per core = round(zone_bytes) given as total bytes / cores (approx, per-core rows printed).
"""
import collections, csv, sys

ops_csv, dev_csv, idx, row_bytes = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
bin_us = float(sys.argv[5]) if len(sys.argv) > 5 else 25.0
rows_of = {}  # optional: "x,y:rows" overrides via env
gen = [r for r in csv.DictReader(open(ops_csv)) if r["OP CODE"].startswith("GenericOp")]
rid = gen[idx]["GLOBAL CALL COUNT"]
f = open(dev_csv)
freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
rd = csv.reader(f)
hdr = [h.strip() for h in next(rd)]
I = {h: i for i, h in enumerate(hdr)}
starts = {}
iv = collections.defaultdict(list)
kstart = []
for r in rd:
    r = [x.strip() for x in r]
    if r[I["run host ID"]] != rid:
        continue
    xy = (int(r[I["core_x"]]), int(r[I["core_y"]]))
    risc, z, t = r[I["RISC processor type"]], r[I["zone name"]], int(r[I["time[cycles since reset]"]])
    key = (xy, risc, z)
    if r[I["type"]] == "ZONE_START":
        starts[key] = t
        if z.endswith("-KERNEL"):
            kstart.append(t)
    elif r[I["type"]] == "ZONE_END" and key in starts:
        iv[key].append((starts.pop(key), t))
t0 = min(kstart)
# rows per core: from the number of cmp_pass1 zones? pass1 runs per block. Use NCRISC rd_read zone count x block rows
# unknown -> estimate from the TRISC_0 cmp_pass1 duration relative to the max (1-row cores take ~half).
p1 = {xy: sum(e - s for s, e in v) for (xy, risc, z), v in iv.items() if risc == "TRISC_0" and z == "cmp_pass1"}
mx = max(p1.values())
rows = {xy: (2 if d > 0.75 * mx else 1) for xy, d in p1.items()}
nb = int(max(e for v in iv.values() for _, e in v) - t0) // int(bin_us * freq) + 1
rbin, wbin = [0.0] * nb, [0.0] * nb


def spread(arr, s, e, byts):
    s, e = (s - t0) / freq, (e - t0) / freq
    if e <= s:
        return
    rate = byts / (e - s)
    b = int(s // bin_us)
    while b * bin_us < e and b < nb:
        lo, hi = max(s, b * bin_us), min(e, (b + 1) * bin_us)
        if hi > lo:
            arr[b] += rate * (hi - lo)
        b += 1


for (xy, risc, z), v in iv.items():
    n = rows.get(xy, 2)
    if risc == "NCRISC" and z == "rd_read":
        for s, e in v:
            spread(rbin, s, e, n * row_bytes / len(v))
    if risc == "TRISC_2" and z == "cmp_pass2":
        for s, e in v:
            spread(wbin, s, e, n * row_bytes / len(v))
print(
    f"op {idx} DEVICE KERNEL {gen[idx]['DEVICE KERNEL DURATION [ns]']} ns; rows per core histogram",
    collections.Counter(rows.values()),
)
print(" t_us   rd_GB/s  wr_GB/s  sum_GB/s")
for b in range(nb):
    r_, w_ = rbin[b] / (bin_us * 1e3), wbin[b] / (bin_us * 1e3)
    print(f"{b*bin_us:5.0f}  {r_:7.1f}  {w_:7.1f}  {r_+w_:7.1f}  " + "#" * int((r_ + w_) / 10))
print(f"total read {sum(rbin)/1e6:.1f} MB write {sum(wbin)/1e6:.1f} MB")
