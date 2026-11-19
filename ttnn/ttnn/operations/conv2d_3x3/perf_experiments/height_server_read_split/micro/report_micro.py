# usage: report_micro.py <profile_log_device.csv[.gz]> <ids file>: per run: kernel span (first start to last end, us)
# and each engine's mb_read zone (core, RISC: start-end us from the run's first kernel start).
import collections, csv, gzip, sys

path, idfile = sys.argv[1], sys.argv[2]
ids = [l.strip() for l in open(idfile) if l.strip()]
f = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
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
nrun = 0
for run in sorted(runs):
    ev = sorted(runs[run], key=lambda e: e[2])
    if not any(e[3] == "mb_read" for e in ev):
        continue
    name = ids[nrun] if nrun < len(ids) else f"run{run}"
    nrun += 1
    t0 = min(e[2] for e in ev if e[3].endswith("-KERNEL") and e[4] == "ZONE_START")
    t1 = max(e[2] for e in ev if e[3].endswith("-KERNEL") and e[4] == "ZONE_END")
    st, zs = {}, []
    for c, risc, t, z, k in ev:
        if z != "mb_read":
            continue
        if k == "ZONE_START":
            st[(c, risc)] = t
        elif (c, risc) in st:
            zs.append((c, risc, (st.pop((c, risc)) - t0) / freq, (t - t0) / freq))
    rd_end = max(b for *_, b in zs)
    rd_start = min(a for _, _, a, _ in zs)
    print(
        f"{name:40s} span {(t1 - t0) / freq:.2f} us  read window {rd_start:.2f}-{rd_end:.2f} = {rd_end - rd_start:.2f} us  "
        + "  ".join(f"{c}{risc[0]}:{a:.2f}-{b:.2f}" for c, risc, a, b in sorted(zs))
    )
