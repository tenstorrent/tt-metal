import sys
sys.path.insert(0, "/tmp"); import initwin
# winpos.py <cyc> <init_elf dir>: absolute start/end cycle of each thread's last INIT window
for t, name in enumerate(("unpack", "math", "pack")):
    r, e, b, d = initwin.window_pcs(f"{sys.argv[2]}/{name}.elf")
    f = open(sys.argv[1]); keys = f.readline().split(); ix = {k: i for i, k in enumerate(keys)}
    pcs = []; prev = None
    for c, line in enumerate(f):
        p = initwin.n(line.split()[ix[f"t{t}_cmt_pc"]])
        if p != prev: pcs.append((c, p)); prev = p
    st = [c for c, p in pcs if p == r]; en = [c for c, p in pcs if p == e]
    s = st[-1]; x = min(c for c in en if c > s)
    print(f"{name:6s} start {s} end {x} length {x - s}")
