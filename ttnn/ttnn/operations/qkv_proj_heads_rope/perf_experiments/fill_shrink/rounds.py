"""Per-round zone timeline (us from the earliest kernel start) for a few cores: python rounds.py <profile_log_device.csv>"""
import collections, csv, sys

path = sys.argv[1]
zones = (
    sys.argv[2].split(",")
    if len(sys.argv) > 2
    else [
        "writer_w_read",
        "writer_w_mcast_send",
        "writer_w_receive",
        "reader_x_receive",
        "reader_x_mcast_send",
        "reader_x_read_issue",
    ]
)
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))[1:]
ev = collections.defaultdict(list)
open_ = {}
t0 = None
for r in rows:
    if len(r) < 12:
        continue
    core = (int(r[1]), int(r[2]))
    zone = r[10].strip()
    typ = r[11].strip()
    t = int(r[5])
    if zone.endswith("-KERNEL") and typ == "ZONE_START":
        t0 = t if t0 is None else min(t0, t)
    if typ == "ZONE_START":
        open_[(core, zone)] = t
    elif typ == "ZONE_END" and (core, zone) in open_:
        ev[(core, zone)].append((open_.pop((core, zone)), t))
cores = sorted({c for c, _ in ev})
sel = [c for c in cores if ev.get((c, "writer_w_read"))][:2] + [c for c in cores if not ev.get((c, "writer_w_read"))][
    :2
]
for c in sel:
    print("core", c)
    for z in zones:
        if ev.get((c, z)):
            print(f"  {z:22}", " ".join(f"{(s-t0)/freq:5.1f}-{(e-t0)/freq:5.1f}" for s, e in ev[(c, z)]))
