"""Per-run zone summary: runzones.py <profile_log_device.csv> <calls.jsonl> <run index>...
Per (zone, risc): median / max over cores of the per-core summed ns, and the median / max per-core
last end (ns from the run's first kernel start)."""
import collections, csv, json, statistics, sys

path, labp = sys.argv[1], sys.argv[2]
idxs = [int(a) for a in sys.argv[3:]]
with open(path) as fh:
    freq = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(fh, skipinitialspace=True))
ns = lambda c: c * 1000.0 / freq
runs = collections.defaultdict(list)
for r in rows:
    runs[r["run host ID"]].append(r)
ours = [k for k in sorted(runs, key=int) if any(r["zone name"] == "reader_coef_share" for r in runs[k])]
labels = [json.loads(l) for l in open(labp)]
res = {}
for i in idxs:
    k = ours[i]
    starts = {}
    dur = collections.defaultdict(lambda: collections.defaultdict(float))
    end = collections.defaultdict(dict)
    t0 = min(int(r["time[cycles since reset]"]) for r in runs[k] if r["zone name"].endswith("-KERNEL"))
    for r in runs[k]:
        t = int(r["time[cycles since reset]"])
        key = (r["core_x"], r["core_y"], r["RISC processor type"], r["zone name"])
        if r["type"] == "ZONE_START":
            starts[key] = t
        elif r["type"] == "ZONE_END" and key in starts:
            z = (r["zone name"], r["RISC processor type"])
            c = (r["core_x"], r["core_y"])
            dur[z][c] += ns(t - starts.pop(key))
            end[z][c] = ns(t - t0)
    res[i] = (dur, end)
zones = sorted(set(z for i in idxs for z in res[i][0]), key=lambda z: (z[1], z[0]))
print(f"{'zone':26s} {'risc':7s} " + " ".join(f"{labels[i]['who']+'#'+str(i):>30s}" for i in idxs))
print(" " * 35 + " ".join(f"{'med':>7s}{'max':>8s}{'endmed':>8s}{'endmax':>7s}" for i in idxs))
for z in zones:
    line = f"{z[0]:26s} {z[1]:7s} "
    for i in idxs:
        d = res[i][0].get(z)
        e = res[i][1].get(z)
        if not d:
            line += " " * 31
            continue
        v = sorted(d.values())
        ev = sorted(e.values())
        line += f"{v[len(v)//2]:7.0f}{v[-1]:8.0f}{ev[len(ev)//2]:8.0f}{ev[-1]:7.0f}"
    print(line)
