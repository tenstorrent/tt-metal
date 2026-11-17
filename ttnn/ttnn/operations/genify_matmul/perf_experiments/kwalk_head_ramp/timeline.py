"""kwalk_head_ramp: K-delivery timeline from profile_log_device.csv (one op run per profiler dump).
Per delivery i: landing time on the reader (end of the i-th rd_mcast_b, R1a; else rd_barrier) relative to
the run's earliest marker, p50 / max over cores; plus the end of cmp_matmul / cmp_epilogue (TRISC_0)."""
import csv, sys, collections, statistics as st

CLK = 1350.0  # cycles per us
rows = list(csv.reader(open(sys.argv[1])))[2:]
by_run = collections.defaultdict(list)
for r in rows:
    by_run[r[7].strip()].append(r)
run = max(by_run, key=lambda k: len(by_run[k]))
rr = by_run[run]
# per-core time base: the earliest *-KERNEL zone start on that core (kernel start)
t0 = {}
for r in rr:
    if r[10].strip().endswith("-KERNEL") and r[11].strip() == "ZONE_START":
        c = (int(r[1]), int(r[2]))
        t0[c] = min(t0.get(c, 1 << 62), int(r[5]))
ends = collections.defaultdict(list)  # (core, risc, zone) -> [end times]
starts = collections.defaultdict(list)
for r in rr:
    c = (int(r[1]), int(r[2]))
    if c not in t0:
        continue
    key = (c, r[3].strip(), r[10].strip())
    (starts if r[11].strip() == "ZONE_START" else ends)[key].append((int(r[5]) - t0[c]) / CLK)
land_zone = "rd_mcast_b" if any(k[2] == "rd_mcast_b" for k in ends) else "rd_barrier"
per_i = collections.defaultdict(list)
for (core, risc, z), ts in ends.items():
    if z == land_zone:
        for i, t in enumerate(sorted(ts)):
            per_i[i].append(t)
out = [f"run {run} landing zone {land_zone}"]
for i in sorted(per_i)[:6]:
    v = per_i[i]
    out.append(f"  delivery {i}: land p50 {st.median(v):7.1f} us  max {max(v):7.1f}  (cores {len(v)})")
for z in ("cmp_matmul", "cmp_epilogue"):
    v = [max(ts) for (c, risc, zz), ts in ends.items() if zz == z and risc.startswith("TRISC_0")]
    if v:
        out.append(f"  {z} end p50 {st.median(v):7.1f} us  max {max(v):7.1f}")
# reader blocked on full CBs
v = [sum(e - s for s, e in zip(sorted(starts[k]), sorted(ends[k]))) for k in ends if k[2] == "rd_reserve"]
if v:
    out.append(f"  rd_reserve sum p50 {st.median(v):7.1f} us")
print("\n".join(out))
