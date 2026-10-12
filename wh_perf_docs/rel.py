import sys
# rel.py <run.cyc>: per thread, the cycles where it starts committing again after >= 200 idle cycles (a park release), and BRISC's last pcs
f = open(sys.argv[1]); keys = f.readline().split(); ix = {k: i for i, k in enumerate(keys)}
last = {t: None for t in ("t0", "t1", "t2", "b")}; lastc = {t: -10**9 for t in last}; ev = []
for c, line in enumerate(f):
    v = line.split()
    for t in last:
        p = v[ix[t + "_cmt_pc"]] if t + "_cmt_pc" in ix else None
        if p is None: continue
        if p != last[t]:
            if c - lastc[t] >= 200 and last[t] is not None:
                ev.append((c, t, c - lastc[t], int(p, 2) if set(p) <= {"0", "1"} else p))
            last[t] = p; lastc[t] = c
print("cycles", c)
for e in ev[-24:]:
    print(e[0], e[1], "idle", e[2], hex(e[3]) if isinstance(e[3], int) else e[3])
