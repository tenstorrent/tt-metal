"""Per-invocation summary of a --profile run: device kernel ns and x-phase zones.

usage: python3 analyze.py <report_dir> case1,case2,...   (2 op invocations per case, the 2nd is reported)
Zone times are ns from the earliest *-KERNEL start of that invocation; "compute cores" = cores running TRISC.
"""
import collections
import csv
import glob
import statistics
import sys

rep = sys.argv[1]
cases = sys.argv[2].split(",")
ops = list(csv.DictReader(open(glob.glob(f"{rep}/ops_perf_results*.csv")[0])))
with open(f"{rep}/profile_log_device.csv") as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))[1:]
by_run = collections.defaultdict(list)
for r in rows:
    by_run[int(r[7])].append(r)
ns = lambda c: c * 1000.0 / freq


def spans(run_rows):
    ev = collections.defaultdict(list)
    for r in run_rows:
        ev[((int(r[1]), int(r[2])), r[3], r[10])].append((int(r[5]), r[11]))
    out = collections.defaultdict(dict)
    for (core, risc, zone), lst in ev.items():
        lst.sort()
        stack, first, last, tot = [], None, None, 0
        for t, typ in lst:
            if typ == "ZONE_START":
                stack.append(t)
            elif typ == "ZONE_END" and stack:
                s = stack.pop()
                tot += t - s
                first = s if first is None else min(first, s)
                last = t if last is None else max(last, t)
        if first is not None:
            out[(risc, zone)][core] = (first, last, tot)
    return out


for i, case in enumerate(cases):
    op = ops[2 * i + 1]
    run = int(op["GLOBAL CALL COUNT"])
    sp = spans(by_run[run])
    t0 = min(v[0] for (risc, z), d in sp.items() if z.endswith("-KERNEL") for v in d.values())
    compute = set(sp.get(("TRISC_0", "TRISC-KERNEL"), {}).keys())

    def end(risc, zone, cores=None):
        d = sp.get((risc, zone), {})
        e = [ns(v[1] - t0) for c, v in d.items() if cores is None or c in cores]
        return (statistics.median(e), max(e)) if e else (float("nan"), float("nan"))

    xr = end("BRISC", "writer_x_receive", compute)
    xp = end("BRISC", "writer_x_phase", compute)
    fk = end("TRISC_0", "compute_first_kblock_wait", compute)
    ws = end("NCRISC", "reader_w_stream", compute)
    yw = end("BRISC", "writer_y_store", compute)
    sends = sorted((ns(v[0] - t0), ns(v[1] - t0)) for v in sp.get(("BRISC", "writer_x_send"), {}).values())
    pf = end("NCRISC", "ncrisc_x_prefetch")
    print(
        f"{case:22} kernel {int(op['DEVICE KERNEL DURATION [ns]']):6d} ns | x_recv_end p50/max {xr[0]:6.0f}/{xr[1]:6.0f}"
        f" x_phase_end {xp[0]:6.0f}/{xp[1]:6.0f} first_kblk {fk[0]:6.0f}/{fk[1]:6.0f} w_end {ws[0]:6.0f}/{ws[1]:6.0f}"
        f" y_end {yw[0]:6.0f}/{yw[1]:6.0f} ncrisc_pf_end_max {pf[1]:6.0f}"
    )
    print("    sends(start-end):", " ".join(f"{a:.0f}-{b:.0f}" for a, b in sends))
