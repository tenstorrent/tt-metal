"""Per-core tail: for the cores that finish last, landing end / compute end / write end (one zones CSV)."""
import csv, collections, sys

rows = list(csv.reader(open(sys.argv[1])))[2:]
by = collections.defaultdict(list)
for r in rows:
    by[(r[0], r[7])].append((int(r[1]), int(r[2]), r[3], r[10], r[11], int(r[5])))
for dev in sorted({k[0] for k in by}):
    last = max((k[1] for k in by if k[0] == dev), key=int)
    ev = by[(dev, last)]
    t0 = min(t for *_, n, ty, t in ev if n.endswith("-KERNEL") and ty == "ZONE_START")
    st, core = {}, collections.defaultdict(lambda: collections.defaultdict(list))
    for x, y, risc, n, ty, t in ev:
        k = (x, y, risc, n)
        if ty == "ZONE_START":
            st.setdefault(k, []).append(t)
        elif ty == "ZONE_END" and st.get(k):
            s = st[k].pop()
            if n == "cmp_head" and risc != "TRISC_0":
                continue
            core[(x, y)][n].append(((s - t0) / 1350, (t - t0) / 1350))
    lst = []
    for c, z in core.items():
        if "wr_head" not in z:
            continue
        lst.append((max(e for _, e in z["wr_head"]), c, z))
    lst.sort(reverse=True)
    print(f"dev {dev}")
    for we, c, z in lst[: int(sys.argv[2]) if len(sys.argv) > 2 else 6]:
        land = sorted(e for _, e in z.get("rd_wait_remote", []))
        cmp_ = sorted((round(s, 1), round(e, 1)) for s, e in z.get("cmp_head", []))
        print(f"  core {c} wr_end {we:5.1f} landings {[round(v,1) for v in land]} cmp {cmp_}")
