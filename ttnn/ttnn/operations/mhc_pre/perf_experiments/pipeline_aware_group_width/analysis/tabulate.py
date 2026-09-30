import sys

for fn in sys.argv[1:]:
    d = {}
    for l in open(fn):
        p = l.split()
        if "us |" not in l:
            continue
        kv = dict(x.split("=") for x in p[4:11])
        d[(p[1], p[3])] = (kv, float(p[11].replace("us", "")), p[13:16])
    print(fn)
    print("| TxC | default (w/B/d) | default us | cand (w/B/d) | cand us | delta |")
    print("|---|---|---|---|---|---|")
    keys = sorted({k[0] for k in d}, key=lambda s: tuple(map(int, s.split("x"))))
    for k in keys:
        (a, ua, sa), (b, ub, sb) = d[(k, "default")], d[(k, "cand")]
        same = (a["w"], a["B"], a["d"]) == (b["w"], b["B"], b["d"])
        print(
            f"| {k} | {a['w']}/{a['B']}/{a['d']} | {ua:.1f} | {b['w']}/{b['B']}/{b['d']}{' (same)' if same else ''} | {ub:.1f} | {(ub/ua-1)*100:+.1f}% |"
        )
