# usage: parse.py ORDER CSV -> per item: median kernel ns of launches 2..12, cores, per-RISC medians
import csv, statistics, sys

order = [l.split() for l in open(sys.argv[1]) if l.strip()]
rows = [
    r
    for r in csv.DictReader(open(sys.argv[2]))
    if r.get("OP TYPE") == "tt_dnn_device" and ("GenericOp" in r["OP CODE"] or "Typecast" in r["OP CODE"])
]
assert len(rows) == 12 * len(order), (len(rows), len(order))
K = "DEVICE KERNEL DURATION [ns]"
for i, (case, var) in enumerate(order):
    g = rows[12 * i + 1 : 12 * i + 12]
    med = statistics.median(float(r[K]) for r in g)

    def risc(n):
        v = [
            float(r[f"DEVICE {n} KERNEL DURATION [ns]"])
            for r in g
            if r.get(f"DEVICE {n} KERNEL DURATION [ns]") not in (None, "", "-")
        ]
        return int(statistics.median(v)) if v else "-"

    print(
        f"{case:<7} {var:<34} {med:>9.0f}  cores={g[0].get('CORE COUNT'):<4} B={risc('BRISC')} N={risc('NCRISC')} T0={risc('TRISC0')} T1={risc('TRISC1')} T2={risc('TRISC2')}"
    )
