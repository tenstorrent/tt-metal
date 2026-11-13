"""Per-port zone breakdown of the last profiled run: zparse.py <csv>. Sender (BRISC) and reader (NCRISC) per core."""
import csv, sys, collections

rows = list(csv.reader(open(sys.argv[1])))[2:]
F = 1350.0
PORT_ZONES = (
    "AG_SEND",
    "AG_P_",
    "AG_R_",
    "AG_ARRIVAL",
    "AG_STEP",
    "AG_LOCAL",
    "AG_INPUT",
    "AG_FINAL",
    "AG_FENCE",
    "AG_READY",
)
for chip in sorted(set(r[0] for r in rows)):
    cr = [r for r in rows if r[0] == chip]
    last = max(int(r[7]) for r in cr)
    sel = sorted([r for r in cr if int(r[7]) == last], key=lambda r: int(r[5]))
    t0 = min(int(r[5]) for r in sel)
    st = collections.defaultdict(list)
    per = collections.defaultdict(lambda: collections.defaultdict(list))  # (x,y,risc) -> name -> [(s,e)]
    for r in sel:
        n = r[10]
        if not n.startswith("AG_"):
            continue
        k = (r[1], r[2], r[3], n)
        if r[11] == "ZONE_START":
            st[k].append(int(r[5]))
        elif st[k]:
            s = st[k].pop()
            per[(r[1], r[2], r[3])][n].append(((s - t0) / F, (int(r[5]) - t0) / F))
    print(f"== chip {chip} run {last}")
    for core, zs in sorted(per.items()):
        if not any(n.startswith(PORT_ZONES) for n in zs) or not (
            ("AG_SEND" in zs) or ("AG_R_RESV" in zs) or ("AG_ARRIVAL_WAIT" in zs)
        ):
            continue
        parts = []
        for n, v in sorted(zs.items(), key=lambda kv: kv[1][0][0]):
            tot = sum(e - s for s, e in v)
            if len(v) == 1:
                parts.append(f"{n}[{v[0][0]:.2f}-{v[0][1]:.2f}]")
            else:
                parts.append(f"{n} n={len(v)} sum={tot:.2f}")
        if "AG_SEND" in zs:
            s, e = zs["AG_SEND"][0]
            npk = len(zs.get("AG_P_ISSUE", []))
            acc = sum(sum(b - a for a, b in zs.get(z, [])) for z in ("AG_P_CBW", "AG_P_ISSUE", "AG_P_FLUSH"))
            parts.append(f"| send {e-s:.2f}us pk={npk} {((e-s)/npk if npk else 0):.3f}us/pk other={e-s-acc:.2f}")
        print(f"  {core}: " + "  ".join(parts))
