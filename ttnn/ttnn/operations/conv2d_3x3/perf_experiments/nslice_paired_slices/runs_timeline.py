import csv, sys, collections

path, runs, core = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], tuple(int(x) for x in sys.argv[3].split(","))
with open(path) as f:
    hdr = f.readline()
    freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rd = csv.reader(f)
    cols = [c.strip() for c in next(rd)]
    ix = {c: i for i, c in enumerate(cols)}
    ev = collections.defaultdict(list)
    for r in rd:
        if len(r) < len(cols):
            continue
        run = int(r[ix["run host ID"]])
        if run not in runs:
            continue
        c = (int(r[ix["core_x"]]), int(r[ix["core_y"]]))
        if c != core:
            continue
        ev[run].append(
            (
                int(r[ix["time[cycles since reset]"]]),
                r[ix["RISC processor type"]].strip(),
                r[ix["zone name"]].strip(),
                r[ix["type"]].strip(),
            )
        )
for run in runs:
    e = sorted(ev[run])
    t0 = min(t for t, *_ in e)
    print(f"=== run {run} core {core}")
    st = {}
    out = []
    for t, risc, z, k in e:
        if risc.startswith("TRISC") and risc != "TRISC_0":
            continue
        key = (risc, z)
        if k == "ZONE_START":
            st.setdefault(key, []).append(t)
        elif k == "ZONE_END" and st.get(key):
            s = st[key].pop()
            if "KERNEL" in z or z.startswith("writer_bias") or z.startswith("reader_fill_reserve"):
                continue
            out.append(((s - t0) / freq, (t - t0) / freq, risc, z))
    for s, e_, risc, z in sorted(out):
        print(f"  {s:7.2f} -> {e_:7.2f} ({e_ - s:6.2f})  {risc:<8} {z}")
