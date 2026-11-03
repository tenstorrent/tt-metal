"""Startup timeline per core (cycles from the core's earliest KERNEL start; same-core clock only).
Usage: python3 -I timeline.py <profile_log_device.csv>
Prints mean / max over cores of: first end of each zone, plus the first-N-instances of a few zones."""
import csv, sys, collections

path = sys.argv[1]
with open(path) as fh:
    fh.readline()
    rows = list(csv.reader(fh))
cols = [c.strip() for c in rows[0]]
ix = {c: i for i, c in enumerate(cols)}
last = max(
    int(r[ix["run host ID"]]) for r in rows[1:] if len(r) >= len(cols) and r[ix["run host ID"]].strip().isdigit()
)
ev = collections.defaultdict(list)  # core -> (t, risc, name, typ)
for r in rows[1:]:
    if len(r) < len(cols) or r[ix["run host ID"]].strip() != str(last):
        continue
    core = (r[ix["core_x"]].strip(), r[ix["core_y"]].strip())
    ev[core].append(
        (
            int(r[ix["time[cycles since reset]"]]),
            r[ix["RISC processor type"]].strip(),
            r[ix["zone name"]].strip(),
            r[ix["type"]].strip(),
        )
    )
stats = collections.defaultdict(list)
for core, es in ev.items():
    es.sort()
    t0 = min(t for t, risc, n, ty in es if n.endswith("-KERNEL") and ty == "ZONE_START")
    seen = collections.Counter()
    starts = {}
    for t, risc, n, ty in es:
        if n.endswith("-FW"):
            continue
        k = (risc, n)
        if ty == "ZONE_START":
            starts[k] = t
        elif ty == "ZONE_END" and k in starts:
            seen[k] += 1
            c = seen[k]
            if c <= 3 or n.endswith("-KERNEL"):
                stats[(risc, n, c, "end")].append(t - t0)
                stats[(risc, n, c, "dur")].append(t - starts[k])
for (risc, n, c, what), v in sorted(stats.items()):
    if what != "end":
        continue
    d = stats[(risc, n, c, "dur")]
    print(
        f"{risc:7s} {n:28s} #{c} end mean={sum(v)/len(v):8.0f} max={max(v):8d}  dur mean={sum(d)/len(d):7.0f} max={max(d):7d}  cores={len(v)}"
    )
