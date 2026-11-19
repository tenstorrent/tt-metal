# usage: percore.py <dev csv.gz> <run idx> : per core (sorted by x, y): kernel end, reader_fill end, writer weight_fetch
# ends (rounds), compute side-round weight wait, first khx_send start.
import collections, csv, gzip, sys

path, idx = sys.argv[1], int(sys.argv[2])
with gzip.open(path, "rt") as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    r = csv.reader(f)
    cols = [c.strip() for c in next(r)]
    ix = {c: i for i, c in enumerate(cols)}
    runs = collections.defaultdict(list)
    for row in r:
        if len(row) < len(cols):
            continue
        runs[int(row[ix["run host ID"]])].append(
            (
                (int(row[ix["core_x"]]), int(row[ix["core_y"]])),
                row[ix["RISC processor type"]].strip(),
                int(row[ix["time[cycles since reset]"]]),
                row[ix["zone name"]].strip(),
                row[ix["type"]].strip(),
            )
        )
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
ev = sorted(runs[big[idx - 1]], key=lambda e: e[2])
kst = {}
for core, risc, t, z, kind in ev:
    if z.endswith("-KERNEL") and kind == "ZONE_START":
        kst[core] = min(kst.get(core, t), t)
zones = collections.defaultdict(list)
stack = collections.defaultdict(list)
for core, risc, t, z, kind in ev:
    if kind == "ZONE_START":
        stack[(core, risc, z)].append(t)
    elif stack[(core, risc, z)]:
        s = stack[(core, risc, z)].pop()
        zones[core].append((risc, (s - kst[core]) / freq, (t - kst[core]) / freq, z))
print("times relative to each core's own first kernel start (us)")
for core in sorted(zones):
    zz = zones[core]

    def ends(name, risc=None):
        return [
            round(e, 2) for rr, s, e, z in sorted(zz, key=lambda v: v[1]) if z == name and (risc is None or rr == risc)
        ]

    def starts(name, risc=None):
        return [
            round(s, 2) for rr, s, e, z in sorted(zz, key=lambda v: v[1]) if z == name and (risc is None or rr == risc)
        ]

    kend = max(e for rr, s, e, z in zz if z.endswith("-KERNEL"))
    if any(z.startswith("server") for _, _, _, z in zz):
        print(
            core,
            "SERVER flush ends",
            ends("server_flush"),
            "ack waits",
            [(round(s, 2), round(e, 2)) for rr, s, e, z in sorted(zz, key=lambda v: v[1]) if z == "server_wait_ack"],
        )
        continue
    print(
        core,
        f"end {kend:.2f}",
        "fill",
        ends("reader_fill"),
        "wfetch",
        ends("writer_weight_fetch"),
        "cww",
        list(zip(starts("compute_kwf_weight_wait", "TRISC_0"), ends("compute_kwf_weight_wait", "TRISC_0"))),
        "khx",
        starts("writer_khx_send"),
    )
