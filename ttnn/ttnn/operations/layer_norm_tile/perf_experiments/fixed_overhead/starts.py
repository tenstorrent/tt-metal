import csv, sys
from collections import defaultdict

path = "/localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv"
f = open(path)
h = f.readline()
freq = float(h.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
st = defaultdict(dict)
t0 = {}
for r in rows[1:]:
    r = [c.strip() for c in r]
    run = r[idx["run host ID"]]
    if r[idx["zone name"]] == "NCRISC-KERNEL" and r[idx["type"]] == "ZONE_START":
        core = (int(r[idx["core_x"]]), int(r[idx["core_y"]]))
        t = int(r[idx["time[cycles since reset]"]])
        st[run][core] = t
for run, d in st.items():
    m = min(d.values())
    print("run", run)
    ys = sorted({c[1] for c in d})
    xs = sorted({c[0] for c in d})
    for y in ys:
        print(f"y={y:2d} " + " ".join(f"{(d[(x,y)]-m)*1000/freq:5.0f}" if (x, y) in d else "    -" for x in xs))
