import sys
# lfsr.py <cyc> <cycle> <n> <thread>: type_rand (branch-type cache LFSR) and commit pc for n cycles from <cycle>
f = open(sys.argv[1]); keys = f.readline().split(); ix = {k: i for i, k in enumerate(keys)}
s, n, t = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
out = []
for c, line in enumerate(f):
    if c < s: continue
    if c >= s + n: break
    v = line.split(); out.append(int(v[ix[f"{t}_type_rand"]], 2))
print(" ".join(f"{x:2d}" for x in out))
