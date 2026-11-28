"""Raw and clock-corrected kernel span per launch from a profile_log_device.csv.
Raw = first KERNEL zone start to last KERNEL zone end over all cores (DEVICE KERNEL DURATION's form);
corrected = the same after subtracting each core's wall-clock offset (clock_offset_map.json, STEP A).
Usage: true_span.py <profile_log_device.csv> [skip_first=1]. Prints medians in ns at 1350 MHz."""
import collections, csv, json, statistics, sys
from pathlib import Path

OFF = json.load(open(Path(__file__).parent / "clock_offset_map.json"))["offset"]
path = sys.argv[1]
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 1
start = collections.defaultdict(dict)
end = collections.defaultdict(dict)
with open(path) as f:
    next(f)
    for r in csv.DictReader(f, skipinitialspace=True):
        if not r["zone name"].endswith("-KERNEL"):
            continue
        core = (int(r["core_x"]), int(r["core_y"]))
        run = int(r["run host ID"])
        t = int(r["time[cycles since reset]"])
        if r["type"] == "ZONE_START":
            start[run][core] = min(t, start[run].get(core, t))
        else:
            end[run][core] = max(t, end[run].get(core, t))
runs = sorted(start)[skip:]
raw, cor, cores = [], [], None
for run in runs:
    s, e = start[run], end[run]
    cores = sorted(s)
    raw.append(max(e.values()) - min(s.values()))
    o = {c: OFF[f"{c[0]},{c[1]}"] for c in s}
    cor.append(max(e[c] - o[c] for c in e) - min(s[c] - o[c] for c in s))
ns = lambda v: v / 1.35
print(
    f"TRUE_SPAN launches={len(runs)} cores={len(cores)} raw_ns={ns(statistics.median(raw)):.0f} corrected_ns={ns(statistics.median(cor)):.0f} cores_xy={cores}"
)
