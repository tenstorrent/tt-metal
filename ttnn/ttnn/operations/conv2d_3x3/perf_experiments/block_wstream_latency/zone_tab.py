# usage: zone_tab.py <device csv> <run_index> [k_cols]: median / max over grid and spare cores of each zone's start and
# end (relative to the core's own BRISC kernel start), for the walk-tail zones. Host-only.
import re, subprocess, sys, statistics

Z = "c.tilize,c.matmul_walk,c.combine,w.store,r.combine_send,w.weight_fetch".split(",")
ZN = (
    ",".join(z.replace("c.", "compute_").replace("w.", "writer_").replace("r.", "reader_") for z in Z) + ",BRISC-KERNEL"
)
out = subprocess.run(
    [sys.executable, __file__.replace("zone_tab.py", "rounds_tl.py"), sys.argv[1], sys.argv[2], ZN],
    capture_output=True,
    text=True,
).stdout
k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
acc = {}
for l in out.splitlines():
    m = re.match(r"\(\s*(\d+),(\d+)\) phys\((\d+), (\d+)\) end ([0-9.]+)\s+(.*)", l)
    if not m:
        continue
    d = {}
    for part in m.group(6).split("|"):
        part = part.strip()
        if part:
            name, *iv = part.split()
            d[name] = [tuple(map(float, x.split("-"))) for x in iv]
    st = d["BRISC-KERNEL"][0][0]
    grp = "grid" if int(m.group(1)) < k else "spare"
    acc.setdefault((grp, "end"), []).append(float(m.group(5)) - st)
    for z in d:
        if z == "BRISC-KERNEL":
            continue
        iv = d[z]
        if z.endswith("weight_fetch"):
            acc.setdefault((grp, z + " last"), []).append(iv[-1][1] - st)
            continue
        acc.setdefault((grp, z + " s"), []).append(iv[0][0] - st)
        acc.setdefault((grp, z + " e"), []).append(iv[-1][1] - st)
for key in sorted(acc, key=lambda k: (k[0], statistics.median(acc[k]))):
    v = acc[key]
    print(f"{key[0]:5} {key[1]:<28} med {statistics.median(v):5.2f} max {max(v):5.2f}")
