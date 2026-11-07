"""Per run: the combine sums' pure cost. For each leader / root, the sum after its LAST slot arrived:
zone end - (last reader_collect end on that core, i.e. when the reader pushed the final slot).
Prints min / median over leaders and roots (TRISC_1 zones). Usage: sumcost.py csv"""
import collections, csv, sys

path = sys.argv[1]
with open(path) as fh:
    freq = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(fh, skipinitialspace=True))
runs = collections.defaultdict(list)
for r in rows:
    runs[int(r.get("run host ID", 0) or 0)].append(r)
for rid in sorted(runs):
    st, z = {}, collections.defaultdict(list)
    for r in runs[rid]:
        core = (int(r["core_x"]), int(r["core_y"]))
        n, risc = r["zone name"], r["RISC processor type"]
        k = (core, risc, n)
        t = int(r["time[cycles since reset]"])
        if r["type"] == "ZONE_START":
            st[k] = t
        elif k in st:
            s = st.pop(k)
            if n.startswith("compute_") and risc != "TRISC_1":
                continue
            z[(core, n)].append((s, t))
    out = {}
    for name in ("compute_lead_combine", "compute_root_combine"):
        vals = []
        for (c, n), v in z.items():
            if n != name:
                continue
            s, e = v[-1]
            col = [ce for (cs, ce) in z.get((c, "reader_collect"), []) if ce <= e]
            ready = max([s] + col)
            vals.append((e - ready) / freq * 1000)
        vals.sort()
        out[name] = vals
    print(
        f"RUN {rid}: "
        + " | ".join(
            f"{k[8:]} after-last-slot ns min {v[0]:.0f} med {v[len(v)//2]:.0f} max {v[-1]:.0f} (n={len(v)})"
            for k, v in out.items()
            if v
        )
    )
