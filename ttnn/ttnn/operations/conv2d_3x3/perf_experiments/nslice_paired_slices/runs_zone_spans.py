import csv, sys, collections

path, runs, zone, col = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3], int(sys.argv[4])
with open(path) as f:
    hdr = f.readline()
    freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rd = csv.reader(f)
    cols = [c.strip() for c in next(rd)]
    ix = {c: i for i, c in enumerate(cols)}
    ev = collections.defaultdict(list)
    t0 = {}
    for r in rd:
        if len(r) < len(cols):
            continue
        run = int(r[ix["run host ID"]])
        if run not in runs:
            continue
        c = (int(r[ix["core_x"]]), int(r[ix["core_y"]]))
        t = int(r[ix["time[cycles since reset]"]])
        t0[run] = min(t0.get(run, t), t)
        if c[0] != col:
            continue
        ev[(run, c)].append((t, r[ix["zone name"]].strip(), r[ix["type"]].strip()))
for run in runs:
    print(f"=== run {run} {zone} (start->end us) column x={col}")
    for (rn, c), e in sorted(ev.items()):
        if rn != run:
            continue
        st = []
        spans = []
        for t, z, k in sorted(e):
            if z != zone:
                continue
            if k == "ZONE_START":
                st.append(t)
            elif st:
                s = st.pop()
                spans.append(f"{(s-t0[run])/freq:5.1f}-{(t-t0[run])/freq:5.1f}")
        print(f"  {c}: " + "  ".join(spans))
