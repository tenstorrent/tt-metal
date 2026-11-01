"""Pivot: cell x variant device-kernel ns from copied ops CSV + profile_log_device.csv + dispatch order json.
Duration = max(*-FW ZONE_END) - min(*-FW ZONE_START) over every Tensix core / RISC-V of the run id, at the CSV's
CHIP_FREQ (the ops CSV column is also printed as a cross-check: col/fw ratio).
usage: table.py <ops.csv> <dev.csv> <order.json> [more triples...]"""
import csv, json, sys, collections

t = collections.defaultdict(lambda: collections.defaultdict(list))
ratio = []
vars_ = []
for i in range(1, len(sys.argv), 3):
    ops = [r for r in csv.DictReader(open(sys.argv[i])) if r["OP CODE"].startswith("GenericOp")]
    order = json.load(open(sys.argv[i + 2]))
    assert len(ops) == len(order), (len(ops), len(order))
    rid = {r["GLOBAL CALL COUNT"]: (lab, int(r["DEVICE KERNEL DURATION [ns]"])) for r, lab in zip(ops, order)}
    f = open(sys.argv[i + 1])
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rd = csv.reader(f)
    hdr = [h.strip() for h in next(rd)]
    I = {h: k for k, h in enumerate(hdr)}
    lo, hi = {}, {}
    for r in rd:
        r = [x.strip() for x in r]
        run = r[I["run host ID"]]
        if run not in rid or not r[I["zone name"]].endswith("-FW"):
            continue
        ts = int(r[I["time[cycles since reset]"]])
        if r[I["type"]] == "ZONE_START":
            lo[run] = min(lo.get(run, ts), ts)
        else:
            hi[run] = max(hi.get(run, ts), ts)
    for run, (lab, col) in rid.items():
        cell, var = lab.split(":")
        ns = round((hi[run] - lo[run]) / freq * 1000)
        t[cell][var].append(ns)
        ratio.append(col / ns)
        if var not in vars_:
            vars_.append(var)
print("ops-CSV/fw ratio min %.3f max %.3f" % (min(ratio), max(ratio)))
print("%-22s" % "cell" + "".join("%12s" % v for v in vars_) + "   best/base")
import statistics

reps = any(len(v) > 1 for d in t.values() for v in d.values())
for cell, dl in t.items():
    d = {k: statistics.median(v) for k, v in dl.items()}  # repeated dispatches: median
    base = d.get("base")
    line = "%-22s" % cell + "".join("%12s" % (round(d[v]) if v in d else "-") for v in vars_)
    if base:
        best = min(v for k, v in d.items() if k != "base")
        line += "   %.3f" % (best / base)
    if reps:
        line += "   " + " ".join("%s=%s" % (k, v) for k, v in dl.items())
    print(line)
