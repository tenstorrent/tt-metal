# usage: zone_hash_check.py : hash every MaybeDeviceZoneScope location of this experiment's kernel copies (and the real
# op's) as the profiler does; report collisions (16-bit zone hashes break the profiler at device close).
import glob
import re
from pathlib import Path

HERE = Path(__file__).parent


def h16(s):
    h = 2166136261
    for c in s.encode():
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return ((h & 0xFFFF) ^ (h >> 16)) & 0xFFFF


bad_total = 0
for d in ["op_base", "op_copy", "grad"]:
    files = glob.glob(str(HERE / d / "kernels" / "*"))
    if not files:
        continue
    locs = set()
    for f in files:
        for i, l in enumerate(open(f), 1):
            m = re.search(r'MaybeDeviceZoneScope\("([^"]+)"\)', l)
            if m:
                locs.add(f"{m.group(1)},{f},{i},KERNEL_PROFILER")
    seen = {}
    for s in locs:
        seen.setdefault(h16(s), []).append(s)
    bad = [v for v in seen.values() if len(v) > 1]
    for v in bad:
        print("COLLIDE", d, v)
    bad_total += len(bad)
    print(d, len(locs), "locations,", len(bad), "collisions")

# The graduation: grad/'s kernels at the real op's paths (the profiler hashes the compiled file path), plus the
# firmware / dispatch zone locations from the last profiler logs, as perf_experiments/ablation_r12/zone_hash_check.py.
root = "/localdev/pmilojevic/workspaces/opgen/"
real_dir = root + "ttnn/ttnn/operations/conv2d_3x3/kernels/"
locs = set()
for f in glob.glob(str(HERE / "grad" / "kernels" / "*")):
    for i, l in enumerate(open(f), 1):
        m = re.search(r'MaybeDeviceZoneScope\("([^"]+)"\)', l)
        if m:
            locs.add(f"{m.group(1)},{real_dir}{Path(f).name},{i},KERNEL_PROFILER")
for log in ["generated/profiler/.logs/new_zone_src_locations.log", "generated/profiler/.logs/zone_src_locations.log"]:
    try:
        for line in open(root + log, errors="replace"):
            p = line.find("'#pragma message: ")
            if p >= 0:
                s = line[p + 18 :].rstrip("\n")[:-1]
                if "/conv2d_3x3/" not in s:
                    locs.add(s)
    except FileNotFoundError:
        pass
seen = {}
for s in locs:
    seen.setdefault(h16(s), []).append(s)
bad = [v for v in seen.values() if len(v) > 1]
for v in bad:
    print("COLLIDE grad@real", v)
print("grad@real", len(locs), "locations,", len(bad), "collisions")
raise SystemExit(1 if bad or bad_total else 0)
