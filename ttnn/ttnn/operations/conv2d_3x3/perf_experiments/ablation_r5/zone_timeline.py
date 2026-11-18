import sys, collections

sys.path.insert(0, "/localdev/pmilojevic/workspaces/opgen/ttnn/ttnn/operations/conv2d_3x3/perf_experiments")
import zone_report as zr

f, run, core = sys.argv[1], int(sys.argv[2]), tuple(int(x) for x in sys.argv[3].split(","))
freq, runs = zr.load(f)
ev = [e for e in runs[run] if e[0] == core]
t0 = min(e[2] for e in ev)
stack = collections.defaultdict(list)
rows = []
for c, risc, t, zone, kind in sorted(ev, key=lambda e: e[2]):
    if kind == "ZONE_START":
        stack[(risc, zone)].append(t)
    elif kind == "ZONE_END" and stack[(risc, zone)]:
        s = stack[(risc, zone)].pop()
        rows.append(((s - t0) / freq, (t - t0) / freq, risc, zone))
for s, e, risc, zone in sorted(rows):
    if risc in ("TRISC_0", "TRISC_2"):
        continue
    print(f"{s:7.2f} {e:7.2f} {e-s:6.2f} {risc:8s} {zone}")
