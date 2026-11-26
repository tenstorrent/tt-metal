"""Dispatch-core timeline across the last replay edge on one chip (needs TT_METAL_DEVICE_PROFILER_DISPATCH=1).
Usage: python3 lg_dispatch.py <csv> [chip] — events (us relative to the previous program's last kernel end)
from 2 us before that end to the next program's first kernel start."""
import csv, sys
from collections import defaultdict

F = 1350.0
path = sys.argv[1]
chip = sys.argv[2] if len(sys.argv) > 2 else "0"
import gzip

rows = [
    r for r in list(csv.reader((gzip.open(path, "rt") if path.endswith(".gz") else open(path))))[2:] if r[0] == chip
]
kz = defaultdict(list)
op = {}
for r in rows:
    if r[10].strip().endswith("-KERNEL") and "CQ" not in r[10]:
        key = (r[1], r[2], r[3])
        if r[11].strip() == "ZONE_START":
            op[key] = int(r[5])
        elif key in op:
            kz[key].append((op.pop(key), int(r[5])))
mx = max(len(v) for v in kz.values())
keys = [k for k, v in kz.items() if len(v) == mx]
prev_end = max(sorted(kz[k])[mx - 2][1] for k in keys)
next_start = min(sorted(kz[k])[mx - 1][0] for k in keys)
print(f"edge gap {(next_start - prev_end) / F:.2f} us")
ev = []
for r in rows:
    if r[10].strip().endswith("-KERNEL"):
        continue
    t = int(r[5])
    if prev_end - 2 * F <= t <= next_start + 0.2 * F:
        ev.append((t, f"{r[1]},{r[2]} {r[3]:7s} {r[10].strip()} {r[11].strip()} {r[6]}"))
for t, s in sorted(ev):
    print(f"{(t - prev_end) / F:8.3f} {s}")

# compact: dispatcher (BRISC on the dispatch core) command zones in the window, with durations
print("--- dispatcher command zones (start us, duration us) ---")
st = {}
disp_xy = None
for t, s in sorted(ev):
    parts = s.split()
    if parts[1] != "BRISC" or "CQ-DISPATCH" not in s or "SUBORDINATE:" in s:
        continue
    name, typ = parts[2], parts[3]
    if typ == "ZONE_START":
        st[name] = t
    elif name in st:
        a = st.pop(name)
        print(f"{(a - prev_end) / F:8.3f} {(t - a) / F:6.3f}  {name.replace('CQ-DISPATCH:4:', '')}")
