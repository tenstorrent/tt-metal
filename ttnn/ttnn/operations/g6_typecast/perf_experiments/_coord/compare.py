import sys, re


def load(f):
    d = {}
    for l in open(f):
        m = re.match(r"(\S+)\s+(\S+)\s+ok=\S+ cores=(\d+) med=([\d.]+) B=(\S+) N=(\S+) T0=(\S+) T1=(\S+) T2=(\S+)", l)
        if m:
            d[m.group(1)] = (int(m.group(3)), float(m.group(4)), *m.groups()[4:])
    return d


op = load(sys.argv[1])
st = load(sys.argv[2])
spec = {}
for l in open("eval/golden_tests/g6_typecast/feature_spec.py"):
    m = re.search(r'_s\("(\w+)", (\([^)]*\)), (\w+), (\w+), (\w+), "(\w+)"', l)
    if m:
        spec[m.group(1)] = " ".join(m.groups()[1:])
loss = 0
for k in spec:
    if k in op and k in st:
        r = st[k][1] / op[k][1]
        loss += r < 1
        print(
            f"{k:6s} {spec[k]:42s} c={op[k][0]:3d} op={op[k][1]:9.0f} stock={st[k][1]:9.0f} speedup={r:5.2f} {'LOSS' if r<0.98 else ('tie' if r<1.02 else '')}  opRISC={op[k][2:]} stRISC={st[k][2:]}"
        )
print("losses(<1):", loss)
