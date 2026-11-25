"""Per chip: ring position (from RW_LOCAL cores) and per-source row landing times (diag zones). Host only.
    python3 src_profile.py zones.csv results/coremap.json"""
import csv, json, sys
from collections import defaultdict

rows = list(csv.DictReader((lambda f: (f.readline(), f)[1])(open(sys.argv[1])), skipinitialspace=True))
cm = json.load(open(sys.argv[2]))
Hl = cm["Hl"]
heads = {tuple(c): hs for c, hs in zip(cm["cores"], cm["heads_of"])}
MHZ = 1350.0
for chip in sorted({r["PCIe slot"] for r in rows}, key=int):
    R = [r for r in rows if r["PCIe slot"] == chip]
    lid = max(int(r["run host ID"]) for r in R)
    R = [r for r in R if int(r["run host ID"]) == lid]
    xs = sorted({int(r["core_x"]) for r in R})
    ys = sorted({int(r["core_y"]) for r in R})
    t0 = min(
        int(r["time[cycles since reset]"]) for r in R if r["zone name"].endswith("KERNEL") and r["type"] == "ZONE_START"
    )
    ev = defaultdict(list)
    for r in R:
        c = (xs.index(int(r["core_x"])), ys.index(int(r["core_y"])))
        if r["type"] == "ZONE_END" and r["zone name"] in ("RW_ROW", "RW_LOCAL"):
            ev[(c, r["zone name"])].append((int(r["time[cycles since reset]"]) - t0) / MHZ)
    # local source = the source common to every core with an RW_LOCAL zone
    cand = None
    for (c, z), v in ev.items():
        if z == "RW_LOCAL":
            s = {h // Hl for h in heads.get(c, [])}
            cand = s if cand is None else cand & s
    pos = sorted(cand)[0] if cand and len(cand) == 1 else None
    prof = defaultdict(lambda: defaultdict(list))  # src -> row t -> landing times (single-remote-source cores)
    for (c, z), v in ev.items():
        if z != "RW_ROW":
            continue
        srcs = sorted({h // Hl for h in heads.get(c, []) if h // Hl != pos})
        if len(srcs) == 1:
            n = sum(1 for h in heads[c] if h // Hl == srcs[0])
            for i, t in enumerate(sorted(v)):
                prof[srcs[0]][i // n].append(t)
    loc = [v[0] for (c, z), v in ev.items() if z == "RW_LOCAL"]
    print(
        f"slot {chip}: pos {pos}  own quarter complete (RW_LOCAL end) min/max {min(loc):.1f}/{max(loc):.1f}"
        if loc
        else f"slot {chip}: pos {pos}"
    )
    for s in sorted(prof):
        print(
            f"   src {s}: row landing max per t " + " ".join(f"t{t}:{max(v):6.1f}" for t, v in sorted(prof[s].items()))
        )
