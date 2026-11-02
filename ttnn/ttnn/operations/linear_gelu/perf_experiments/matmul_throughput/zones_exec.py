"""Per-execution compute zones (TRISC_1 = math) of the last run in profile_log_device.csv, for the busiest cores.

Usage: python zones_exec.py [run_index_from_end=1] [risc=TRISC_1]
Prints per core: kernel span, compute_fill, then each compute_matmul_gelu execution (us), for the 3 cores with the
largest compute_matmul_gelu total.
"""
import csv
import sys
from collections import defaultdict

path = sys.argv[3] if len(sys.argv) > 3 else str(__import__("pathlib").Path(__file__).parent / "last_profile_log.csv")
back = int(sys.argv[1]) if len(sys.argv) > 1 else 1
want_risc = sys.argv[2] if len(sys.argv) > 2 else "TRISC_1"
with open(path) as f:
    first = f.readline()
    freq = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(f, skipinitialspace=True))
run_ids = sorted({int(r["run host ID"]) for r in rows})
run_id = run_ids[-back]
open_ = {}
execs = defaultdict(list)
for r in rows:
    if int(r["run host ID"]) != run_id or r["RISC processor type"] != want_risc:
        continue
    core = (int(r["core_x"]), int(r["core_y"]))
    key = (core, r["zone name"])
    t = int(r["time[cycles since reset]"])
    if r["type"] == "ZONE_START":
        open_[key] = t
    elif r["type"] == "ZONE_END" and key in open_:
        s = open_.pop(key)
        execs[core].append((r["zone name"], s, t))
tot = {
    c: sum(e - s for z, s, e in v if z in ("compute_matmul_gelu", "reader_x_mcast", "writer_drain"))
    for c, v in execs.items()
}
print(f"run {run_id} ({len(run_ids)} runs), {want_risc}, {freq} MHz, cores={len(execs)}")
for core in sorted(tot, key=tot.get, reverse=True)[:3]:
    v = sorted(execs[core], key=lambda z: z[1])
    t0 = min(s for _, s, _ in v)
    parts = [f"{z[:12]}:{(e - s) / freq:.1f}@{(s - t0) / freq:.1f}" for z, s, e in v if not z.startswith(want_risc)]
    print(core, f"mm_total={tot[core] / freq:.1f}us", " ".join(parts))
