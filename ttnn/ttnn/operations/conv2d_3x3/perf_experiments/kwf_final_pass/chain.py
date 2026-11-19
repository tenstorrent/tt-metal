# usage: chain.py <csv.gz> <idx:label:tiles>...: the critical core's final-pass chain (last pass) from the kfz_* zones:
# drain = pack handoff zone, pack setup = handoff end -> first final pack section start, tiles = section start -> last
# pack, tail = last pack -> TRISC end; unpack ACC wait and unpack setup.
import collections, csv, gzip, sys

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    hdr = f.readline()
    freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = csv.reader(f)
    cols = [c.strip() for c in next(rows)]
    ix = {c: i for i, c in enumerate(cols)}
    runs = collections.defaultdict(list)
    for r in rows:
        if len(r) < len(cols):
            continue
        runs[int(r[ix["run host ID"]])].append(
            (
                (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                r[ix["RISC processor type"]].strip(),
                int(r[ix["time[cycles since reset]"]]),
                r[ix["zone name"]].strip(),
                r[ix["type"]].strip(),
            )
        )
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
print(
    f"{'case':22s} {'drain':>6s} {'p.setup':>7s} {'tiles':>6s} {'cyc/t':>6s} {'tail':>5s} {'total':>6s} | {'u.accw':>6s} {'u.init':>6s}  (us, zoned, critical core, last pass)"
)
for spec in sys.argv[2:]:
    idx, label, tiles = spec.split(":")
    tiles = int(tiles)
    ev = sorted(runs[big[int(idx) - 1]], key=lambda e: e[2])
    st = collections.defaultdict(list)
    sp = collections.defaultdict(list)
    for core, risc, t, z, kind in ev:
        if kind == "ZONE_START":
            st[(core, risc, z)].append(t)
        elif st[(core, risc, z)]:
            sp[(core, risc, z)].append((st[(core, risc, z)].pop(), t))
    kend = {}
    for (core, risc, z), v in sp.items():
        if z == "TRISC-KERNEL":
            kend[core] = max(kend.get(core, 0), max(e for s, e in v))
    c = max(kend, key=kend.get)
    g = lambda risc, z: sp[(c, risc, z)][-1]
    h0, h1 = g("TRISC_2", "kfz_handoff")
    first = sp[(c, "TRISC_2", "kfz_sec0")][-1][0] if sp.get((c, "TRISC_2", "kfz_sec0")) else g("TRISC_2", "kfz_rest")[0]
    last = g("TRISC_2", "kfz_rest")[1]
    ua = g("TRISC_0", "kfz_accw")
    ui = g("TRISC_0", "kfz_init")
    us = lambda x: x / freq
    print(
        f"{label:22s} {us(h1-h0):6.2f} {us(first-h1):7.2f} {us(last-first):6.2f} {(last-first)/tiles*1350/freq:6.1f} {us(kend[c]-last):5.2f} {us(kend[c]-h0):6.2f} | {us(ua[1]-ua[0]):6.2f} {us(ui[1]-ui[0]):6.2f}"
    )
