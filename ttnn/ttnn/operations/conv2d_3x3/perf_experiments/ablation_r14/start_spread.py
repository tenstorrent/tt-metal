import collections, csv, statistics, sys

path = sys.argv[1]
names = "stem1 stem2 stem3 res2 res3.0 res3.x res4.0 res4.x res5.0 res5.1 res5.2 aspp1 aspp2 aspp3 sdr3f0 idr3f0 sdr3f1 sdr2f0 idr2f0 sr2f1 ir2f1 ch1".split()
with open(path) as f:
    hdr = f.readline()
    freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = csv.reader(f)
    cols = [c.strip() for c in next(rows)]
    ix = {c: i for i, c in enumerate(cols)}
    runs = collections.defaultdict(list)
    for r in rows:
        if len(r) < len(cols):
            continue
        z = r[ix["zone name"]].strip()
        if not z.endswith("-KERNEL"):
            continue
        runs[int(r[ix["run host ID"]])].append(
            (
                (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                int(r[ix["time[cycles since reset]"]]),
                r[ix["type"]].strip(),
            )
        )
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
tot = 0
for i, k in enumerate(big):
    st = {}
    en = {}
    for c, t, kind in runs[k]:
        if kind == "ZONE_START":
            st[c] = min(st.get(c, t), t)
        else:
            en[c] = max(en.get(c, t), t)
    t0 = min(st.values())
    D = (max(en.values()) - t0) / freq
    own = max((en[c] - st[c]) / freq for c in st)
    crit = max(en, key=lambda c: en[c])
    s50 = statistics.median((v - t0) / freq for v in st.values())
    print(
        f"{names[i] if i<len(names) else i:7s} D {D:6.2f}  max own span {own:6.2f}  start-spread cost {D-own:5.2f}  crit {crit} starts {(st[crit]-t0)/freq:.2f}  p50 start {s50:.2f}"
    )
    tot += D - own
print("sum", round(tot, 2))
