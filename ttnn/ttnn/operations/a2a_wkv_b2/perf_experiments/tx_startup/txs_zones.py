"""tx_startup: per fabric_tx core (all chips), start/end (us from the replay instance's first kernel start) of
the tx zones in the last trace replay instance. Usage: python3 txs_zones.py <csv> [--replay i]"""
import csv, sys, bisect
from collections import defaultdict

FREQ = 1350.0
path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
which = int(sys.argv[sys.argv.index("--replay") + 1]) if "--replay" in sys.argv else -1
rows = list(csv.reader(open(path)))[2:]
kst = defaultdict(list)
for r in rows:
    if r[10].strip().endswith("-KERNEL") and r[11].strip() == "ZONE_START":
        kst[r[0]].append(int(r[5]))
inst = {}
for chip, ts in kst.items():
    ts.sort()
    starts = [ts[0]]
    for a, b in zip(ts, ts[1:]):
        if b - a > 5 * FREQ:
            starts.append(b)
    inst[chip] = starts
sel = {c: (len(s) - 1 if which < 0 else which) for c, s in inst.items()}
z = defaultdict(lambda: [None, None])  # (chip, x, y, zone) -> [start, end]
kend = defaultdict(int)
for r in rows:
    chip, x, y, risc, _, t = r[:6]
    t = int(t)
    i = bisect.bisect_right(inst[chip], t) - 1
    if i != sel[chip]:
        continue
    zone, typ = r[10].strip(), r[11].strip()
    t0 = inst[chip][sel[chip]]
    if zone.endswith("-KERNEL") and typ == "ZONE_END":
        kend[chip] = max(kend[chip], t)
    if not zone.startswith("tx_"):
        continue
    k = (chip, int(x), int(y), zone)
    if typ == "ZONE_START" and z[k][0] is None:
        z[k][0] = (t - t0) / FREQ
    elif typ == "ZONE_END":
        z[k][1] = (t - t0) / FREQ
chips = sorted(inst, key=int)
print(
    "span per chip (us):",
    " ".join(f"{c}:{(kend[c]-inst[c][sel[c]])/FREQ:.1f}" for c in chips),
    f" max {max((kend[c]-inst[c][sel[c]])/FREQ for c in chips):.1f}",
)
cores = sorted({(x, y) for (_, x, y, _) in z})
zones = [
    "tx_writer_open",
    "tx_writer_arm",
    "tx_writer_addr_table",
    "tx_writer_table_sync",
    "tx_reader_addr_table",
    "tx_writer_await_ready",
    "tx_writer_stream",
    "tx_writer_close",
    "tx_writer_await_done",
    "tx_reader_stream",
]
print(f"{'core':8s} {'zone':22s} start[min/max]   end[min/max]  (over chips)")
for x, y in cores:
    for zn in zones:
        v = [z[(c, x, y, zn)] for c in chips if (c, x, y, zn) in z and z[(c, x, y, zn)][1] is not None]
        if not v:
            continue
        s = [a for a, _ in v if a is not None]
        e = [b for _, b in v]
        print(f"({x},{y})  {zn:22s} {min(s):5.1f}/{max(s):5.1f}    {min(e):5.1f}/{max(e):5.1f}")
