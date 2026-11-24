"""Per port core (last call of each device): sender span, time the sender waited on the reader (psd_cbwait), reader span,
time the reader waited on its DRAM barriers (prd_barrier) and on ring space (prd_resv). usage: port_zones.py <csv>"""
import collections, csv, sys

rows = list(csv.reader(open(sys.argv[1])))[2:]
by = collections.defaultdict(list)
for r in rows:
    by[(r[0], r[7])].append((int(r[1]), int(r[2]), r[3], r[10], r[11], int(r[5])))
tot = collections.defaultdict(float)
for dev in sorted({k[0] for k in by}):
    last = max((k[1] for k in by if k[0] == dev), key=int)
    ev = by[(dev, last)]
    t0 = min(t for *_, n, ty, t in ev if n.endswith("-KERNEL") and ty == "ZONE_START")
    t1 = max(t for *_, n, ty, t in ev if n.endswith("-KERNEL") and ty == "ZONE_END")
    print(f"dev {dev}: kernel span {(t1 - t0) / 1350:.2f} us")
    starts, z = {}, collections.defaultdict(list)
    for x, y, risc, n, ty, t in ev:
        k = (x, y, risc, n)
        if ty == "ZONE_START":
            starts.setdefault(k, []).append(t)
        elif ty == "ZONE_END" and starts.get(k):
            s = starts[k].pop()
            z[(x, y, n)].append(((s - t0) / 1350, (t - t0) / 1350))
    ports = sorted({(x, y) for (x, y, n) in z if n == "psd_send"})
    for p in ports:
        g = lambda n: z.get((p[0], p[1], n), [])
        send = g("psd_send")
        prd = g("prd_all")
        cw = g("psd_cbwait")
        bar = g("prd_barrier")
        rs = g("prd_resv")
        s_cw = sum(e - s for s, e in cw)
        late = [(s, e - s) for s, e in cw if e - s > 0.3]
        print(
            f"  port {p}: send {send[0][0]:6.2f}-{send[0][1]:6.2f}  cbwait sum {s_cw:5.2f} (n {len(cw)}, >0.3us: "
            + ", ".join(f"{s:.1f}+{d:.2f}" for s, d in late)
            + f")  reader {prd[0][0]:6.2f}-{prd[0][1]:6.2f}  barrier sum {sum(e - s for s, e in bar):5.2f}"
            f" max {max([e - s for s, e in bar] or [0]):4.2f}  resv sum {sum(e - s for s, e in rs):5.2f}"
        )
        tot["cbwait"] += s_cw
        tot["ports"] += 1
print(f"mean cbwait per port {tot['cbwait'] / max(1, tot['ports']):.2f} us")
