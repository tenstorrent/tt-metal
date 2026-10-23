"""Per-core zone timeline for one (device, run host id) of generated/profiler/.logs/profile_log_device.csv.
usage: core_timeline.py DEV RUN [x,y ...]   (no cores: list slowest / fastest BRISC kernel ends). us from the chip-run t0."""
import csv, sys, collections

path = "generated/profiler/.logs/profile_log_device.csv"
DEV = int(sys.argv[1])
RUN = int(sys.argv[2])
cores = sys.argv[3:]
with open(path) as fh:
    hdr = fh.readline()
    mhz = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rd = csv.reader(fh)
    next(rd)
    ev = [r for r in rd if len(r) > 12 and int(r[0]) == DEV and int(r[7]) == RUN]
t0 = min(int(r[5]) for r in ev if r[10].strip().endswith("-KERNEL") and r[11].strip() == "ZONE_START")
open_ = {}
out = collections.defaultdict(list)
for r in sorted(ev, key=lambda r: int(r[5])):
    x, y, risc, t, n, ty = int(r[1]), int(r[2]), r[3].strip(), int(r[5]), r[10].strip(), r[11].strip()
    k = (x, y, risc, n)
    if ty == "ZONE_START":
        open_.setdefault(k, []).append(t)
    elif ty == "ZONE_END" and open_.get(k):
        s = open_[k].pop()
        out[(x, y)].append((risc, n, (s - t0) / mhz, (t - t0) / mhz))
if not cores:
    # summarize end of BRISC kernel per core
    ends = sorted(
        (
            (max(e for r_, n, s, e in v if n == "BRISC-KERNEL"), c)
            for c, v in out.items()
            if any(n == "BRISC-KERNEL" for _, n, _, _ in v)
        ),
        reverse=True,
    )
    print("slowest BRISC ends:", [(round(e, 2), c) for e, c in ends[:12]])
    print("fastest:", [(round(e, 2), c) for e, c in ends[-5:]])
    sys.exit()
for c in cores:
    x, y = map(int, c.split(","))
    print("== core", c)
    for risc, n, s, e in sorted(out[(x, y)], key=lambda z: z[2]):
        if n.endswith("-FW"):
            continue
        print(f"  {risc:8s}{n:18s}{s:7.2f}{e:7.2f}  ({e-s:.2f})")
