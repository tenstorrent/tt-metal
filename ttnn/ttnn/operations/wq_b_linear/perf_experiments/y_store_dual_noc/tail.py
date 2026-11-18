"""Per run of profile_log_device.csv: y-store tail = last kernel end - last pack end (TRISC_2), and y-store zones.

Per core: tail_core = max(BRISC, NCRISC kernel end) - TRISC_2 kernel end. Prints max/p50 over cores, the op end
(max over cores of every RISC's kernel end, relative to the earliest kernel start) and the store zone p50/max.
"""
import collections
import csv
import statistics
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
labels = sys.argv[2].split(",") if len(sys.argv) > 2 else None
with open(path) as f:
    header = f.readline()
    freq = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))[1:]
ns = lambda c: c * 1000.0 / freq
runs = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    runs[int(r[7])][((int(r[1]), int(r[2])), r[3], r[10])].append((int(r[5]), r[11]))
print(
    f"{'run':>6} {'label':8} {'op_end':>7} {'pack_end':>8} {'tail_max':>8} {'tail_p50':>8} {'wst_p50':>7} {'wst_max':>7} {'rst_p50':>7} {'rst_max':>7}"
)
for i, run in enumerate(sorted(runs)):
    ev = runs[run]
    span = {}
    for (core, risc, zone), lst in ev.items():
        lst.sort()
        st = [t for t, typ in lst if typ == "ZONE_START"]
        en = [t for t, typ in lst if typ == "ZONE_END"]
        if st and en:
            span[(core, risc, zone)] = (min(st), max(en))
    t0 = min(v[0] for (c, r, z), v in span.items() if z.endswith("-KERNEL"))
    kend = collections.defaultdict(dict)
    for (c, r, z), v in span.items():
        if z.endswith("-KERNEL"):
            kend[c][r] = v[1]
    op_end = max(max(d.values()) for d in kend.values())
    tails, packs = [], []
    for c, d in kend.items():
        if "TRISC_2" in d:
            packs.append(d["TRISC_2"])
            tails.append(max(d.get("BRISC", 0), d.get("NCRISC", 0)) - d["TRISC_2"])

    def zone(name):
        d = [v[1] - v[0] for (c, r, z), v in span.items() if z == name]
        return (ns(statistics.median(d)), ns(max(d))) if d else (0, 0)

    w, rd = zone("writer_y_store"), zone("reader_y_store")
    lab = labels[i] if labels and i < len(labels) else ""
    print(
        f"{run:6d} {lab:8} {ns(op_end - t0):7.0f} {ns(max(packs) - t0):8.0f} {ns(max(tails)):8.0f} {ns(statistics.median(tails)):8.0f} {w[0]:7.0f} {w[1]:7.0f} {rd[0]:7.0f} {rd[1]:7.0f}"
    )
