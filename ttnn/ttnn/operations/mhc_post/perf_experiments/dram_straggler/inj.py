"""inj.py <single-run csv>: grid of injector flags (I) by reader_issue duration, with reader_barrier sum and body end."""
import csv, collections, sys

lines = open(sys.argv[1]).readlines()
freq = float(lines[0].split("CHIP_FREQ[MHz]:")[1].split(",")[0])
rows = list(csv.DictReader(lines[1:], skipinitialspace=True))
st = {}
iss = collections.defaultdict(list)
bar = collections.defaultdict(float)
be = {}
t0 = None
for r in rows:
    k = (r["core_x"], r["core_y"], r["RISC processor type"], r["zone name"])
    t = int(r["time[cycles since reset]"])
    if r["type"] == "ZONE_START":
        st[k] = t
        if r["zone name"].endswith("-FW"):
            t0 = t if t0 is None else min(t0, t)
    elif k in st:
        dd = (t - st.pop(k)) / freq
        if r["zone name"] == "reader_issue":
            iss[k[:2]].append(dd)
        if r["zone name"] == "reader_barrier":
            bar[k[:2]] += dd
        if r["zone name"] == "compute_mix" and k[2] == "TRISC_0":
            be[k[:2]] = t
xs = sorted({int(c[0]) for c in iss})
ys = sorted({int(c[1]) for c in iss})
print("cell = I(injector)/. barrier_sum_us body_end_us")
print("   " + "".join(f"{x:>13d}" for x in xs))
for y in ys:
    out = []
    for x in xs:
        c = (str(x), str(y))
        if c not in iss:
            out.append(" " * 13)
            continue
        inj = "I" if max(iss[c][:1]) > 0.62 else "."
        out.append(f"{inj}{bar[c]:6.1f}{(be[c]-t0)/freq:6.1f}")
    print(f"{y:3d}" + "".join(out))
