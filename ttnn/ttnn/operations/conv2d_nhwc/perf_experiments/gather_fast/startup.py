"""Per program (last by default): first own K-block gather latency and net gather work.
Usage: python3 -I startup.py [csv]"""
import csv, sys, collections

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
with open(path) as fh:
    fh.readline()
    rows = list(csv.reader(fh))
cols = [c.strip() for c in rows[0]]
ix = {c: i for i, c in enumerate(cols)}
prog = collections.defaultdict(list)
for r in rows[1:]:
    if len(r) >= len(cols):
        prog[r[ix["run host ID"]].strip()].append(r)
pid = sorted(prog, key=lambda k: int(k) if k.isdigit() else 0)[-1]
st = collections.defaultdict(list)
zones = collections.defaultdict(list)  # (core, risc, name) -> [(t0, t1)]
kstart = {}
for r in prog[pid]:
    core = (int(r[ix["core_x"]]), int(r[ix["core_y"]]))
    risc = r[ix["RISC processor type"]].strip()
    name = r[ix["zone name"]].strip()
    t = int(r[ix["time[cycles since reset]"]])
    key = (core, risc, name)
    if r[ix["type"]].strip() == "ZONE_START":
        st[key].append(t)
    elif st[key]:
        t0 = st[key].pop()
        zones[key].append((t0, t))
        if name.endswith("-KERNEL"):
            kstart[(core, risc)] = t0
t_start = min(kstart.values())
cores = sorted(set(c for c, _ in kstart))
first_g, gwork, first_gather_end = [], [], {}
for c in cores:
    g = sorted(zones.get((c, "NCRISC", "reader_gather_kblock"), []))
    rr = zones.get((c, "NCRISC", "reader_rm_reserve"), [])
    if not g:
        continue
    first_gather_end[c] = g[0][1] - kstart[(c, "NCRISC")]
    first_g.append(g[0][1] - g[0][0])
    gwork.append(sum(b - a for a, b in g) - sum(b - a for a, b in rr))
xs = sorted(set(c[0] for c in first_gather_end))
print(f"program {pid}: cores={len(cores)}")
print(f"  first own K-block gather dur: max={max(first_g)} mean={sum(first_g)/len(first_g):.0f}")
print(f"  net gather work/core (kblock - rm_reserve): mean={sum(gwork)/len(gwork):.0f} max={max(gwork)}")
x0 = xs[0]
e0 = [v for c, v in first_gather_end.items() if c[0] == x0]
print(
    f"  slice-0 column (x={x0}) first gather end since own NCRISC kernel start: max={max(e0)} mean={sum(e0)/len(e0):.0f}"
)
print(
    "  slice-0 column first gather end per grid row:",
    " ".join(f"{v}" for c, v in sorted(first_gather_end.items(), key=lambda kv: kv[0][1]) if c[0] == x0),
)
