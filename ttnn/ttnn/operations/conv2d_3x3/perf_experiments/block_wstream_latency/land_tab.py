# usage: land_tab.py <device csv> <run_index> [k_cols]: per round, the median / max landing (writer_weight_fetch end,
# relative to each core's own kernel start) over the grid members and over the spare members. Host-only.
import re, subprocess, sys, statistics

out = subprocess.run(
    [
        sys.executable,
        __file__.replace("land_tab.py", "rounds_tl.py"),
        sys.argv[1],
        sys.argv[2],
        "writer_weight_fetch,BRISC-KERNEL",
    ],
    capture_output=True,
    text=True,
).stdout
k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
g, s, ge, se = {}, {}, [], []
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
    tgt = g if int(m.group(1)) < k else s
    (ge if int(m.group(1)) < k else se).append(float(m.group(5)) - st)
    for r, (a, b) in enumerate(d["w.weight_fetch"]):
        tgt.setdefault(r, []).append(b - st)
print("round  grid med/max   spare med/max   (us after own kernel start)")
for r in sorted(g):
    print(
        f"  r{r}   {statistics.median(g[r]):5.2f} {max(g[r]):5.2f}    {statistics.median(s[r]):5.2f} {max(s[r]):5.2f}"
    )
print(f"  end  {statistics.median(ge):5.2f} {max(ge):5.2f}    {statistics.median(se):5.2f} {max(se):5.2f}")
