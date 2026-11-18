# usage: khx_waits.py <profile_log_device.csv> <run_index (1-based among runs with >= 8 cores)> [zone ...]
# Per core: start/end (us, relative to the run's earliest kernel start) of the named zones on TRISC_0 / BRISC, sorted by
# the core's kernel end; prints the 8 latest-ending cores.
import collections, csv, sys

path, idx = sys.argv[1], int(sys.argv[2])
zones = sys.argv[3:] or ["compute_kwf_halo_wait", "writer_khx_send", "compute_kwf_final"]
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
t0 = min(e[2] for e in ev if e[3].endswith("-KERNEL"))
us = lambda c: round((c - t0) / freq, 2)
open_, spans, kend = {}, collections.defaultdict(list), {}
for core, risc, t, z, kind in ev:
    if z.endswith("-KERNEL") and kind == "ZONE_END" and risc.startswith("TRISC"):
        kend[core] = max(kend.get(core, 0), us(t))
    if z in zones and risc in ("TRISC_0", "BRISC", "NCRISC"):
        key = (core, risc, z)
        if kind == "ZONE_START":
            open_[key] = t
        elif key in open_:
            spans[core].append(
                (
                    z.replace("compute_kwf_", "c.").replace("writer_khx_", "w.").replace("reader_kwf_", "r."),
                    us(open_.pop(key)),
                    us(t),
                )
            )
for core in sorted(kend, key=lambda c: -kend[c])[:8]:
    print(core, "end", kend[core], " ".join(f"{z}[{a}-{b}]" for z, a, b in sorted(spans[core], key=lambda s: s[1])))
