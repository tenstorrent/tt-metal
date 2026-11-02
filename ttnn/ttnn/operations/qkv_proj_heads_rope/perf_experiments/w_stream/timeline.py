"""Per-core zone timeline (us from first kernel start). Usage: timeline.py csv risc zone[,zone] [core filter x:y,...]"""
import collections, csv, sys

path, risc_f, zones = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
cores_f = {tuple(int(v) for v in c.split(":")) for c in sys.argv[4].split(",")} if len(sys.argv) > 4 else None
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))[1:]
t0 = None
open_, ev = {}, collections.defaultdict(list)
kend = {}
for r in rows:
    if len(r) < 12:
        continue
    core = (int(r[1]), int(r[2]))
    risc, t, zone, typ = r[3].strip(), int(r[5]), r[10].strip(), r[11].strip()
    if zone.endswith("-KERNEL") and typ == "ZONE_START":
        t0 = t if t0 is None else min(t0, t)
    if zone.endswith("-KERNEL") and typ == "ZONE_END":
        kend[core] = max(kend.get(core, 0), t)
    key = (core, risc, zone)
    if typ == "ZONE_START":
        open_[key] = t
    elif typ == "ZONE_END" and key in open_:
        ev[(core, risc)].append((open_.pop(key), t, zone))
for (core, risc), items in sorted(ev.items()):
    if risc != risc_f or (cores_f and core not in cores_f):
        continue
    sel = [(s, e, z) for s, e, z in sorted(items) if z in zones]
    if not sel:
        continue
    print(
        core,
        f"end={(kend[core]-t0)/freq:.1f}",
        " ".join(f"{z.split('_')[-1][:4]}[{(s-t0)/freq:.1f}+{(e-s)/freq:.2f}]" for s, e, z in sel),
    )
