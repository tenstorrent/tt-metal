import csv, sys, statistics
from collections import defaultdict

path, want = sys.argv[1], sys.argv[2]
with open(path) as f:
    header = f.readline()
    freq = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
open_ = {}
ev = defaultdict(list)
for r in rows[1:]:
    r = [c.strip() for c in r]
    if r[idx["run host ID"]] != want:
        continue
    core = (int(r[idx["core_x"]]), int(r[idx["core_y"]]))
    risc = r[idx["RISC processor type"]]
    z = r[idx["zone name"]]
    typ = r[idx["type"]]
    t = int(r[idx["time[cycles since reset]"]])
    k = (core, risc, z)
    if typ == "ZONE_START":
        open_[k] = t
    elif typ == "ZONE_END" and k in open_:
        ev[core].append((risc, z, open_.pop(k), t))
t0 = min(s for c in ev for (_, z, s, _) in ev[c] if z.endswith("KERNEL"))
us = lambda t: (t - t0) / freq


def pct(vals):
    vals = sorted(vals)
    n = len(vals)
    return " ".join(f"{vals[min(int(q*n), n-1)]:5.1f}" for q in (0, 0.25, 0.5, 0.75, 1))


for nrows in (1, 2):
    cs = [c for c in ev if sum(1 for e in ev[c] if e[1] == "compute_center" and e[0] == "TRISC_1") == nrows]
    if not cs:
        continue
    print(f"--- {nrows}-row cores: {len(cs)}   (min p25 p50 p75 max)")

    def zone(c, name, which, k=0, risc=None):
        es = sorted([e for e in ev[c] if e[1] == name and (risc is None or e[0] == risc)], key=lambda e: e[2])
        return us(es[k][which]) if len(es) > k else None

    stats = {
        "res0 land (BR residual end)": [zone(c, "writer_residual_block", 3) for c in cs],
        "x0 land (NC reader_block end)": [zone(c, "reader_block", 3) for c in cs],
        "stats end row0": [zone(c, "compute_add_and_stats", 3, 0, "TRISC_1") for c in cs],
        "compute end": [max(us(e[3]) for e in ev[c] if e[1] == "TRISC-KERNEL") for c in cs],
        "store last start": [zone(c, "writer_store_block", 2, nrows - 1) for c in cs],
        "kernel end": [max(us(e[3]) for e in ev[c] if e[1].endswith("KERNEL")) for c in cs],
    }
    for k, v in stats.items():
        v = [x for x in v if x is not None]
        if v:
            print(f"  {k:32s} {pct(v)}")
