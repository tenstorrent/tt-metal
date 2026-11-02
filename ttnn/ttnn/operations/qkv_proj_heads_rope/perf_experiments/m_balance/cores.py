"""Per-core zone sums / ends (us) for the FIRST generic op in a device csv, grouped by compute_matmul sum
(bucketed). Usage: cores.py <csv> [zone,...]"""
import collections, csv, statistics, sys

path = sys.argv[1]
zones = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "TRISC_0:compute_matmul,TRISC_1:compute_matmul,TRISC_2:compute_matmul,TRISC_1:compute_last_k_epilogue,writer_w_receive,writer_w_read,writer_w_mcast_send,reader_x_receive,reader_x_mcast_send,reader_x_read_barrier,writer_store,BRISC-KERNEL,NCRISC-KERNEL,TRISC_1:TRISC-KERNEL"
).split(",")
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))[1:]
open_, tot, first, last = {}, collections.defaultdict(float), {}, {}
t0 = None
seen_end = set()
op = 0
for r in rows:
    if len(r) < 12:
        continue
    core = (int(r[1]), int(r[2]))
    risc, t, zone, typ = r[3].strip(), int(r[5]), r[10].strip(), r[11].strip()
    if zone == "BRISC-KERNEL" and typ == "ZONE_START":
        if t0 is None:
            t0 = t
    if t0 is not None and t - t0 > 300 * freq:  # first op only
        continue
    z = f"{risc}:{zone}" if risc.startswith("TRISC") else zone
    key = (core, z)
    if typ == "ZONE_START":
        open_[key] = t
        first.setdefault(key, t)
    elif typ == "ZONE_END" and key in open_:
        tot[key] += t - open_.pop(key)
        last[key] = t
cores = sorted({k[0] for k in tot})


def g(c, z):
    return tot.get((c, z), 0) / freq


def e(c, z):
    return (last[(c, z)] - t0) / freq if (c, z) in last else 0


buckets = collections.defaultdict(list)
for c in cores:
    buckets[round(g(c, "TRISC_1:compute_matmul"))].append(c)
print("op end", max(e(c, "BRISC-KERNEL") for c in cores))
for b in sorted(buckets):
    cs = buckets[b]
    print(f"--- matmul_sum~{b} us: {len(cs)} cores")
    for z in zones:
        sums = [g(c, z) for c in cs]
        ends = [e(c, z) for c in cs]
        print(
            f"  {z:26} sum med {statistics.median(sums):6.1f} max {max(sums):6.1f}  end med {statistics.median(ends):6.1f} max {max(ends):6.1f}"
        )
