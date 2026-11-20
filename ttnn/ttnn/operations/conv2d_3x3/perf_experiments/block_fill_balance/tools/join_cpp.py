# usage: join_cpp.py <cpp_device_perf_report.csv> <runs/TAG.jsonl>: join device kernel durations (call order) with the
# harness records (the last N); prints case, variant, ns, ratio vs baseline of the same case, sha match vs baseline.
import csv, json, sys, collections

rows = list(csv.DictReader(open(sys.argv[1])))
recs = [json.loads(l) for l in open(sys.argv[2])]
recs = recs[len(recs) - len(rows) :]
base = {}
out = collections.defaultdict(list)
for rec, row in zip(recs, rows):
    ns = int(float(row["DEVICE KERNEL DURATION [ns]"]))
    out[rec["case"]].append((rec["variant"], ns, rec["out_sha"], rec["pcc"]))
for case, lst in out.items():
    b = [x for x in lst if x[0] == "baseline"]
    bns = sorted(x[1] for x in b)[len(b) // 2] if b else None
    bsha = b[0][2] if b else None
    for v, ns, sha, pcc in lst:
        print(
            f"{case:24s} {v:12s} {ns:8d} {ns / bns if bns else 0:6.3f}x sha {'==' if sha == bsha else '!='} pcc {pcc:.6f}"
        )
