"""Summarize per-stage zones per program from profile_log_device.csv + ops CSV.
usage: zones.py [labels comma list] [ops_perf_results.csv] [profile_log_device.csv]
Labels are matched to GenericOp rows in execution order. Without paths: newest ops CSV + generated/profiler/.logs.
Concurrent agents share generated/profiler: copy both files into your own dir IN THE SAME shell command as the run
and pass them here."""
import csv, sys, glob, os, collections, statistics as st

ROOT = "/localdev/adamjanovic/2026_11_01/1349_adamjanovic_ab-tilize-run-scoreboard-catalog4/clones/rms_norm_run1/tt-metal/generated/profiler"
labels = sys.argv[1].split(",") if len(sys.argv) > 1 else []
ops = (
    sys.argv[2]
    if len(sys.argv) > 2
    else sorted(glob.glob(ROOT + "/reports/*/ops_perf_results_*.csv"), key=os.path.getmtime)[-1]
)
rows = [r for r in csv.DictReader(open(ops))]
gen = [r for r in rows if r["OP CODE"].startswith("GenericOp")]
f = open(sys.argv[3] if len(sys.argv) > 3 else ROOT + "/.logs/profile_log_device.csv")
head = f.readline()
freq = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
rd = csv.reader(f)
hdr = [h.strip() for h in next(rd)]
I = {h: i for i, h in enumerate(hdr)}
ev = collections.defaultdict(list)  # (runid, core, risc) -> list
for r in rd:
    r = [x.strip() for x in r]
    ev[r[I["run host ID"]]].append(r)


def summarize(runid):
    starts = {}
    dur = collections.defaultdict(lambda: collections.defaultdict(float))
    cnt = collections.Counter()
    span = {}
    for r in ev[runid]:
        key = (r[I["core_x"]], r[I["core_y"]], r[I["RISC processor type"]])
        z = r[I["zone name"]]
        t = int(r[I["time[cycles since reset]"]])
        cnt[key] += 1
        if r[I["type"]] == "ZONE_START":
            starts[(key, z)] = t
        elif r[I["type"]] == "ZONE_END" and (key, z) in starts:
            d = t - starts.pop((key, z))
            dur[(key[2], z)][key[:2]] += d
            if z.endswith("-KERNEL"):
                span[key] = d
    out = {}
    for (risc, z), per in dur.items():
        v = list(per.values())
        out[(risc, z)] = (max(v) / freq * 1000, st.median(v) / freq * 1000, len(v))
    return out, max(cnt.values()) if cnt else 0


runids = sorted(ev, key=int)
# map generic op rows to runids by GLOBAL CALL COUNT
for i, r in enumerate(gen):
    rid = r["GLOBAL CALL COUNT"]
    lab = labels[i] if i < len(labels) else rid
    print("=== %s  DEVICE KERNEL %s ns  cores=%s" % (lab, r["DEVICE KERNEL DURATION [ns]"], r["CORE COUNT"]))
    if rid not in ev:
        print("   (no zone data for run id", rid, ")")
        continue
    out, mx = summarize(rid)
    print("   max markers on one (core,RISC):", mx)
    for (risc, z), (mxv, p50, n) in sorted(out.items(), key=lambda kv: (kv[0][0], -kv[1][0])):
        if z.endswith("-FW"):
            continue
        print("   %-7s %-20s max=%9.0f p50=%9.0f ns  cores=%d" % (risc, z, mxv, p50, n))
