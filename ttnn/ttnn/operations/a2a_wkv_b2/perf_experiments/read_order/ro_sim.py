"""read_order: placement model. Per worker position (logical x, y) the 5 block publish times (compute_matmul end,
TRISC_2) are taken from a profile (they depend on the core's position, not its head: measured). For a placement
(RO_CORES string), each stream is a single server at RATE us/packet (1 packet per block), starting at START;
link-1 streams also carry the relayed antipode packets of the other side (same publish profile, +HOP us).
Prints the predicted end of each stream. Usage: ro_sim.py profile.csv [placement-spec ...] (spec as ro_place.py
or 'base')"""
import csv, sys, bisect
from collections import defaultdict
import ro_place

F = 1350.0
RATE, START, HOP = 0.275, 13.0, 1.0
path = sys.argv[1]
rows = list(csv.reader(open(path)))[2:]
kst = defaultdict(list)
for r in rows:
    if r[10].strip().endswith("-KERNEL") and r[11].strip() == "ZONE_START":
        kst[r[0]].append(int(r[5]))
inst = {}
for c, ts in kst.items():
    ts.sort()
    s = [ts[0]]
    for a, b in zip(ts, ts[1:]):
        if b - a > 5 * F:
            s.append(b)
    inst[c] = s
ends = defaultdict(list)
for r in rows:
    c = r[0]
    t = int(r[5])
    i = bisect.bisect_right(inst[c], t) - 1
    if i != len(inst[c]) - 1:
        continue
    if r[10].strip() == "compute_matmul" and r[3].strip() == "TRISC_2" and r[11].strip() == "ZONE_END":
        ends[(int(r[1]), int(r[2]))].append((t - inst[c][-1]) / F)
pos = defaultdict(list)  # (phys x, phys y) -> list over chips of 5 sorted times
for (x, y), ts in ends.items():
    pass
per = defaultdict(list)
for r in rows:
    pass
# re-key per chip
byc = defaultdict(list)
for r in rows:
    c = r[0]
    t = int(r[5])
    i = bisect.bisect_right(inst[c], t) - 1
    if i != len(inst[c]) - 1:
        continue
    if r[10].strip() == "compute_matmul" and r[3].strip() == "TRISC_2" and r[11].strip() == "ZONE_END":
        byc[(int(r[1]), int(r[2]))].append((c, (t - inst[c][-1]) / F))
xs = sorted({x for x, y in byc})
ys = sorted({y for x, y in byc})
xs = [x for x in xs if sum(1 for (a, b) in byc if a == x) >= 8]
assert len(ys) >= 8
ys = [y for y in ys if sum(1 for (a, b) in byc if b == y) >= 12][:8]
xs = xs[:12]


def pub(lx, ly):
    """mean over chips of the 5 sorted publish times at logical (lx, ly)"""
    v = byc[(xs[lx], ys[ly])]
    chips = defaultdict(list)
    for c, t in v:
        chips[c].append(t)
    arr = [sorted(t)[:5] for t in chips.values() if len(t) >= 5]
    return [sum(a[j] for a in arr) / len(arr) for j in range(5)]


P = {(lx, ly): pub(lx, ly) for lx in range(12) for ly in range(8)}


def serve(times):
    t = START
    for p in sorted(times):
        t = max(t, p) + RATE
    return t


def predict(cores):
    cls = defaultdict(list)
    for i, c in enumerate(cores):
        cls[c].extend(P[(i % 12, i // 12)])
    if "X" in cls:
        x = cls.pop("X")
        cls["A"] = x[::2]
        cls["B"] = x[1::2]  # approx
    out = {"P(x12,l0)": serve(cls["P"]), "M(x14,l0)": serve(cls["M"])}
    relayB = [t + HOP for t in cls["B"]]
    relayA = [t + HOP for t in cls["A"]]
    out["A(x13,l1)"] = serve(cls["A"] + relayA)  # relay of the other side's +1 antipode (same placement)
    out["B(x15,l1)"] = serve(cls["B"] + relayB)
    out["L_last"] = max(cls["L"]) if cls["L"] else 0
    return out


for spec in sys.argv[2:]:
    if spec == "base":
        cores = "L" * 24 + "A" * 12 + "B" * 12 + "P" * 24 + "M" * 24
    elif "|" in spec:
        cores = ro_place.build(spec)
    else:
        cores = spec
    o = predict(cores)
    print(f"{spec[:60]:60s} " + " ".join(f"{k}={v:5.1f}" for k, v in o.items()) + f"  max={max(o.values()):5.1f}")
