import sys
# wincnt.py <cyc> <start> <len>: per thread mispredicts (cycles with mp=1), code-line fetches, last commit cycle in the window
f = open(sys.argv[1]); keys = f.readline().split(); ix = {k: i for i, k in enumerate(keys)}
s, L = int(sys.argv[2]), int(sys.argv[3]); mp = {t: 0 for t in "012"}; ic = dict(mp); last = dict(mp); prev = {}
for c, line in enumerate(f):
    if c < s: continue
    if c >= s + L: break
    v = line.split()
    for t in "012":
        mp[t] += v[ix[f"t{t}_mp"]] == "1"; ic[t] += v[ix[f"t{t}_ic_rden"]] == "1"
        p = v[ix[f"t{t}_cmt_pc"]]
        if prev.get(t) != p: last[t] = c - s; prev[t] = p
print("mispredict cycles", mp, "| code-line fetches", ic, "| last commit (rel)", last)
