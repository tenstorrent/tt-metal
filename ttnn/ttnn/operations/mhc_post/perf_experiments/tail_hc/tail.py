"""Root-tail timeline per program (run host ID) in profile_log_device.csv. Compute zones from TRISC_1 only.
Per run: kernel end, last writer_ship_partial end, and for the critical root (latest writer_hc_out end):
ready = when its combined stats are in DEST (min of compute_root_combine end, compute_hc start),
compute_hc duration, hc_out end; 'post' = kernel_end - ready (the post-combine tail),
'tail' = kernel_end - last_ship. Usage: tail.py [csv]"""
import collections, csv, sys

path = next((a for a in sys.argv[1:] if not a.startswith("--")), "generated/profiler/.logs/profile_log_device.csv")
with open(path) as fh:
    freq = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(fh, skipinitialspace=True))
runs = collections.defaultdict(list)
for r in rows:
    runs[int(r.get("run host ID", 0) or 0)].append(r)
for rid in sorted(runs):
    starts, z, t0 = {}, collections.defaultdict(list), None
    for r in runs[rid]:
        core = (int(r["core_x"]), int(r["core_y"]))
        n, risc = r["zone name"], r["RISC processor type"]
        key = (core, risc, n)
        t = int(r["time[cycles since reset]"])
        if r["type"] == "ZONE_START":
            starts[key] = t
            if n.endswith("-FW"):
                t0 = t if t0 is None else min(t0, t)
        elif r["type"] == "ZONE_END" and key in starts:
            s = starts.pop(key)
            if n.startswith("compute_") and risc != "TRISC_1":
                continue
            z[(core, n)].append((s, t))
    if not any(n == "writer_ship_partial" for (c, n) in z):
        continue
    us = lambda c: (c - t0) / freq  # noqa: E731
    ship = max(v[-1][1] for (c, n), v in z.items() if n == "writer_ship_partial")
    kend = max(e for (c, n), v in z.items() if n.endswith("-FW") for s, e in v)
    hco = {c: v[-1][1] for (c, n), v in z.items() if n == "writer_hc_out"}
    if not hco:  # stats mode: root = latest compute_root_combine
        hco = {c: v[-1][1] for (c, n), v in z.items() if n == "compute_root_combine"}
    root = max(hco, key=hco.get)
    rend = z[(root, "compute_root_combine")][-1][1] if (root, "compute_root_combine") in z else None
    hc = z.get((root, "compute_hc"), [None])[-1]
    ready = min(x for x in (rend, hc[0] if hc else None) if x is not None)
    hcd = (hc[1] - hc[0]) / freq if hc else 0.0
    print(
        f"RUN {rid}: kernel_end {us(kend):6.2f} last_ship {us(ship):6.2f} root{root} ready {us(ready):6.2f}"
        f" hc {hcd:4.2f} hc_out_end {us(hco[root]):6.2f} | post(kend-ready) {us(kend) - us(ready):4.2f}"
        f" tail(kend-ship) {us(kend) - us(ship):4.2f} us"
    )
