"""Per-Tensix-core zone maps for one GenericOp run (physical NoC coordinates).

usage: percore.py <ops.csv> <profile_log_device.csv> <generic-op index> [zone ...]
Prints, per zone, a grid (rows = physical y, columns = physical x) of the zone's summed duration in us, plus
the kernel END time of every RISC-V relative to the earliest kernel start of the op (who finishes last).
"""
import collections
import csv
import sys

ops_csv, dev_csv, idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
zones = sys.argv[4:] or ["rd_read", "wr_store"]
gen = [r for r in csv.DictReader(open(ops_csv)) if r["OP CODE"].startswith("GenericOp")]
rid = gen[idx]["GLOBAL CALL COUNT"]
print("op", idx, "run id", rid, "DEVICE KERNEL", gen[idx]["DEVICE KERNEL DURATION [ns]"], "ns")
f = open(dev_csv)
freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
rd = csv.reader(f)
hdr = [h.strip() for h in next(rd)]
I = {h: i for i, h in enumerate(hdr)}
starts, dur = {}, collections.defaultdict(float)
kstart, kend = {}, {}
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
            kstart[(xy, risc)] = t
    elif r[I["type"]] == "ZONE_END" and key in starts:
        dur[key] += t - starts.pop(key)
        if z.endswith("-KERNEL"):
            kend[(xy, risc)] = t
t0 = min(kstart.values())
xs = sorted({xy[0] for xy, _ in kstart})
ys = sorted({xy[1] for xy, _ in kstart})


def grid(title, val):
    print(f"--- {title} (us)")
    print("y\\x " + "".join(f"{x:>6}" for x in xs))
    for y in ys:
        print(f"{y:>3} " + "".join(f"{val(( x, y)):6.0f}" if val((x, y)) is not None else "     ." for x in xs))
    vals = [val((x, y)) for x in xs for y in ys if val((x, y)) is not None]
    vals.sort()
    if vals:
        print(f"    min {vals[0]:.0f} p50 {vals[len(vals)//2]:.0f} max {vals[-1]:.0f}")


for z in zones:
    for risc in ("NCRISC", "BRISC", "TRISC_0", "TRISC_1", "TRISC_2"):
        if any(k[1] == risc and k[2] == z for k in dur):
            grid(f"{risc} {z} duration", lambda xy: dur[(xy, risc, z)] / freq if (xy, risc, z) in dur else None)
for risc in ("NCRISC", "BRISC", "TRISC_2"):
    grid(
        f"{risc} kernel END (from op start)", lambda xy: (kend[(xy, risc)] - t0) / freq if (xy, risc) in kend else None
    )
