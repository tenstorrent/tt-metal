import csv, sys
from collections import defaultdict

path = "/localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv"
f = open(path)
h = f.readline()
freq = float(h.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
want = sys.argv[1] if len(sys.argv) > 1 else "reader_affine_send"
ev = defaultdict(list)
op = {}
t0 = {}
for r in rows[1:]:
    r = [c.strip() for c in r]
    run = r[idx["run host ID"]]
    core = (r[idx["core_x"]], r[idx["core_y"]])
    risc = r[idx["RISC processor type"]]
    z = r[idx["zone name"]]
    t = int(r[idx["time[cycles since reset]"]])
    t0[run] = min(t0.get(run, t), t)
    k = (run, core, risc, z)
    if r[idx["type"]] == "ZONE_START":
        op[k] = t
    elif k in op:
        ev[(run, core)].append((op.pop(k), t, risc, z))
for (run, core), es in ev.items():
    if any(z == want for *_, z in es):
        print("run", run, "core", core)
        for s, e, risc, z in sorted(es):
            print(
                f"  {risc:8s} {z:28s} {(s-t0[run])*1000/freq:7.0f} -> {(e-t0[run])*1000/freq:7.0f} ({(e-s)*1000/freq:6.0f})"
            )
