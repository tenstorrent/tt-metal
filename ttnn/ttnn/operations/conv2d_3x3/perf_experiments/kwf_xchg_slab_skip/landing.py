# usage: landing.py <profile_log_device.csv> <run_index>: per core, round-0 landing (end of the first
# compute_kwf_weight_wait on TRISC_0), reader fill end, writer fill end (last writer_fill / writer_fill_early), the first
# side-round start and the kernel end; min / median / max over cores (us from the run's earliest kernel start).
import collections, csv, statistics, sys

path, idx = sys.argv[1], int(sys.argv[2])
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
ev = sorted(runs[big[idx - 1]], key=lambda e: e[2])
t0 = min(e[2] for e in ev if e[3].endswith("-KERNEL") and e[4] == "ZONE_START")
us = lambda c: (c - t0) / freq
r0, rf, wf, ke = {}, {}, {}, {}
for core, risc, t, z, kind in ev:
    if kind != "ZONE_END":
        continue
    if z == "compute_kwf_weight_wait" and risc == "TRISC_0" and core not in r0:
        r0[core] = us(t)
    elif z == "reader_fill" and risc == "NCRISC":
        rf[core] = max(rf.get(core, 0), us(t))
    elif z in ("writer_fill", "writer_fill_early") and risc == "BRISC":
        wf[core] = max(wf.get(core, 0), us(t))
    elif z.endswith("-KERNEL"):
        ke[core] = max(ke.get(core, 0), us(t))
crit = max(ke, key=ke.get)


def st(d):
    v = sorted(d.values())
    return f"{v[0]:.2f}/{statistics.median(v):.2f}/{v[-1]:.2f}" if v else "-"


print(
    f"run {idx}: round0 landing {st(r0)}  reader fill end {st(rf)}  writer fill end {st(wf)}  kernel end {st(ke)}"
    f"  | critical core {crit}: r0 {r0.get(crit, 0):.2f} rfill {rf.get(crit, 0):.2f} wfill {wf.get(crit, 0):.2f} end {ke[crit]:.2f}"
)
