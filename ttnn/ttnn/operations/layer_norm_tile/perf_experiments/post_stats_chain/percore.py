"""Per-core compute-chain zones (TRISC_1 sums, us) and phase ends for one run of a whole-op device CSV.
usage: percore.py <dev.csv> [run_index]   -> slowest 6 cores by BRISC end + chain medians over heavy / light cores"""
import csv, statistics, sys
from collections import defaultdict

path = sys.argv[1]
sel = int(sys.argv[2]) if len(sys.argv) > 2 else 0
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
runs, open_ = [], {}
dur = defaultdict(lambda: defaultdict(float))
cnt = defaultdict(lambda: defaultdict(int))
end = defaultdict(dict)
t0 = {}
for r in rows[1:]:
    r = [c.strip() for c in r]
    run = r[idx["run host ID"]]
    if run not in runs:
        runs.append(run)
    if run != runs[min(sel, len(runs) - 1)] and len(runs) > sel:
        continue
    core = (int(r[idx["core_x"]]), int(r[idx["core_y"]]))
    risc, z, t = r[idx["RISC processor type"]], r[idx["zone name"]], int(r[idx["time[cycles since reset]"]])
    t0[run] = min(t0.get(run, t), t)
    k = (core, risc, z)
    if r[idx["type"]] == "ZONE_START":
        open_[k] = t
    elif k in open_:
        s = open_.pop(k)
        if risc == "TRISC_1":
            dur[core][z] += (t - s) / freq
            cnt[core][z] += 1
        if z.endswith("KERNEL"):
            end[core][risc] = t
run = runs[sel]
base = t0[run]
chain = ["compute_center", "compute_row_stats", "compute_affine_scale", "compute_normalize", "compute_affine_shift"]


def us(t):
    return (t - base) / freq


cores = sorted(end, key=lambda c: -end[c].get("BRISC", 0))
print(f"run {run}: slowest cores by BRISC end (us), TRISC_1 chain sums (us), center count")
for c in cores[:6]:
    e = end[c]
    print(
        f"  {c} brisc_end {us(e.get('BRISC',0)):6.2f} trisc1_end {us(e.get('TRISC_1',0)):6.2f} "
        + " ".join(f"{z[8:]}={dur[c][z]:.2f}" for z in chain + ["compute_add_and_stats"])
        + f" n={cnt[c]['compute_center']}"
    )
for n in (1, 2):
    grp = [c for c in dur if cnt[c]["compute_center"] == n]
    if grp:
        tot = [sum(dur[c][z] for z in chain) for c in grp]
        print(
            f"  cores with {n} blocks: {len(grp)}, chain median {statistics.median(tot):.2f} us, "
            f"trisc1_end median {statistics.median(us(end[c]['TRISC_1']) for c in grp):.2f}"
        )
