# usage: zone_hash_check.py : hash every MaybeDeviceZoneScope location of this experiment's op copies as the profiler
# does (ablation_r14/zone_hash_check.py over op_base/ and op_copy/); report collisions within each copy.
import glob
import re
import sys
from pathlib import Path


def h16(s):
    h = 2166136261
    for c in s.encode():
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return ((h & 0xFFFF) ^ (h >> 16)) & 0xFFFF


here = Path(__file__).resolve().parent
bad_any = False
# grad@op: grad/'s kernels hashed at the real op's kernel path (as they compile once graduation.patch is applied).
op_kernels = here.parents[1] / "kernels"
for sub in ("op_base", "op_copy", "grad", "grad@op"):
    src_dir = here / sub.split("@")[0] / "kernels"
    if not src_dir.exists():
        continue
    locs = set()
    for f in glob.glob(str(src_dir / "*")):
        name = str(op_kernels / Path(f).name) if sub.endswith("@op") else f
        for i, l in enumerate(open(f), 1):
            m = re.search(r'MaybeDeviceZoneScope\("([^"]+)"\)', l)
            if m:
                locs.add(f"{m.group(1)},{name},{i},KERNEL_PROFILER")
    seen = {}
    for s in locs:
        seen.setdefault(h16(s), []).append(s)
    bad = [v for v in seen.values() if len(v) > 1]
    for v in bad:
        print("COLLIDE", sub, v)
    bad_any |= bool(bad)
    print(sub, len(locs), "locations,", len(bad), "collisions")
sys.exit(1 if bad_any else 0)
