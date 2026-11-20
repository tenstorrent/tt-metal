# usage: lands.py <profile_log_device.csv> <run_index (1-based among runs with >= 8 cores)> [rows y,...]
# Per core: the reader's batch landing times (ends of reader_hf_land, in slab order), the tap build's kernel-row ends
# (compute_ts_kh on TRISC_0), walk start and kernel end, us after the run's earliest kernel start. Sorted by (x, y).
import collections, csv, sys

path, idx = sys.argv[1], int(sys.argv[2])
rows_want = {int(v) for v in sys.argv[3].split(",")} if len(sys.argv) > 3 else None
with open(path) as f:
    hdr = f.readline()
    freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rd = csv.reader(f)
    cols = [c.strip() for c in next(rd)]
    ix = {c: i for i, c in enumerate(cols)}
    runs = collections.defaultdict(list)
    for r in rd:
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
us = lambda v: (v - t0) / freq
lands = collections.defaultdict(list)
kh = collections.defaultdict(list)
first = {}
last = {}
for core, risc, t, z, kind in ev:
    key = (core, risc, z)
    if kind == "ZONE_START":
        first.setdefault(key, t)
    else:
        last[key] = t
        if z == "reader_hf_land":
            lands[core].append(us(t))
        if z == "compute_ts_kh" and risc == "TRISC_0":
            kh[core].append(us(t))
cores = sorted({c for c, *_ in ev}, key=lambda c: (c[0], c[1]))
print("core      fill_start  lands(batch ends)                         kh ends            walk   end")
for c in cores:
    if rows_want and c[1] not in rows_want:
        continue
    fs = first.get((c, "NCRISC", "reader_fill"))
    ws = first.get((c, "TRISC_0", "compute_matmul_walk"))
    ke = last.get((c, "BRISC", "BRISC-KERNEL"))
    print(
        f"{str(c):9s} {us(fs) if fs else 0:5.1f}     "
        + " ".join(f"{v:4.1f}" for v in lands[c]).ljust(40)
        + "  "
        + " ".join(f"{v:4.1f}" for v in kh[c]).ljust(16)
        + f"  {us(ws) if ws else 0:5.1f} {us(ke) if ke else 0:5.1f}"
    )
