# usage: startup.py <profile_log_device.csv> [run_index ...]: per run, the FW -> KERNEL start latency per RISC and the
# kernel start spread (us, relative to the run's earliest *-KERNEL start). DEVICE KERNEL DURATION counts the spread.
import collections, csv, statistics, sys

path = sys.argv[1]
want = [int(v) for v in sys.argv[2:]]
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
for i, k in enumerate(big, 1):
    if want and i not in want:
        continue
    st = {}
    en = {}
    for core, risc, t, z, kind in runs[k]:
        key = (core, risc, z)
        if kind == "ZONE_START":
            st[key] = min(st.get(key, t), t)
        else:
            en[key] = max(en.get(key, t), t)
    ks = {k2: v for k2, v in st.items() if k2[2].endswith("-KERNEL")}
    t0 = min(ks.values())
    tend = max(v for k2, v in en.items() if k2[2].endswith("-KERNEL"))
    print(f"run {i} ({k}): duration {(tend - t0) / freq:.2f} us")
    for risc in ("BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"):
        fw = {c: v for (c, r, z), v in st.items() if r == risc and z.endswith("-FW")}
        kk = {c: v for (c, r, z), v in st.items() if r == risc and z.endswith("-KERNEL")}
        lat = [(kk[c] - fw[c]) / freq for c in kk if c in fw]
        kst = [(v - t0) / freq for v in kk.values()]
        fws = [(v - t0) / freq for v in fw.values()]
        if not lat:
            continue
        print(
            f"  {risc:8s} FW start {min(fws):6.2f}..{max(fws):6.2f}  FW->KERNEL p50 {statistics.median(lat):.2f} max {max(lat):.2f}"
            f"  KERNEL start {min(kst):.2f}..{max(kst):.2f} p50 {statistics.median(kst):.2f}  cores {len(kk)}"
        )
