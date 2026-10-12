import sys
# streamdiff.py <A.cyc> <A start> <B.cyc> <B start> <length>: per signal, the change events in [start, start+length) relative to start;
# prints the first events where A and B differ, per thread commit stream, plus packer/xbar summaries
SIG = ["t0_cmt_pc", "t1_cmt_pc", "t2_cmt_pc", "b_cmt_pc", "t0_ic_rden", "t1_ic_rden", "t2_ic_rden", "t0_mp", "t1_mp", "t2_mp",
       "xb_ready", "p0_l1_ready", "p1_l1_ready", "p2_l1_ready", "p3_l1_ready", "t0_l1_rden", "t1_l1_rden", "t2_l1_rden", "t0_ibuf_ready", "t1_ibuf_ready", "t2_ibuf_ready"]
def ev(path, s, L):
    f = open(path); keys = f.readline().split(); ix = {k: i for i, k in enumerate(keys)}
    out = {k: [] for k in SIG if k in ix}; prev = {}
    for c, line in enumerate(f):
        if c < s - 1: continue
        if c >= s + L: break
        v = line.split()
        for k in out:
            x = v[ix[k]]
            if prev.get(k) != x and c >= s:
                out[k].append((c - s, x))
            prev[k] = x
    return out
A = ev(sys.argv[1], int(sys.argv[2]), int(sys.argv[5])); B = ev(sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
for k in A:
    a, b = A[k], B[k]
    i = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
    if i is None and len(a) == len(b):
        print(f"{k:14s} identical ({len(a)} changes)"); continue
    i = i if i is not None else min(len(a), len(b))
    fa = a[i] if i < len(a) else None; fb = b[i] if i < len(b) else None
    def h(e): return None if e is None else (e[0], hex(int(e[1], 2)) if set(e[1]) <= {"0", "1"} and len(e[1]) > 4 else e[1])
    print(f"{k:14s} first difference at change #{i}: A {h(fa)}  B {h(fb)}  (changes A {len(a)} B {len(b)})")
