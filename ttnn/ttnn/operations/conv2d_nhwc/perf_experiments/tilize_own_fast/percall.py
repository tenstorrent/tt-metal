"""Per-call durations of given zones on one RISC, for a few cores. Usage: percall.py csv risc zone [zone...]"""
import csv, sys, collections

path, risc = sys.argv[1], sys.argv[2]
zones = set(sys.argv[3:])
with open(path) as fh:
    fh.readline()
    rows = list(csv.reader(fh))
cols = [c.strip() for c in rows[0]]
ix = {c: i for i, c in enumerate(cols)}
st = {}
out = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows[1:]:
    if len(r) < len(cols) or r[ix["RISC processor type"]].strip() != risc:
        continue
    name = r[ix["zone name"]].strip()
    if name not in zones:
        continue
    core = (int(r[ix["core_x"]]), int(r[ix["core_y"]]))
    t = int(r[ix["time[cycles since reset]"]])
    if r[ix["type"]].strip() == "ZONE_START":
        st[(core, name)] = t
    else:
        out[name][core].append(t - st[(core, name)])
for name, d in out.items():
    cores = sorted(d)
    n = len(d[cores[0]])
    mean = [sum(d[c][i] for c in cores if i < len(d[c])) / len(cores) for i in range(n)]
    print(name, "mean per call:", [int(x) for x in mean])
