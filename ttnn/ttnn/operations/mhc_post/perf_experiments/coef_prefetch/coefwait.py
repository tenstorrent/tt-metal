"""Per run: slowest cores (kernel end) with their summed TRISC_0 compute_wait_coef / compute_reserve_out (TRISC_2)
and count of cores with coef wait > 2 us.  usage: python coefwait.py profile_log_device.csv run_id [run_id ...]"""
import csv, sys
from collections import defaultdict

path, runs = sys.argv[1], set(sys.argv[2:])
MHZ = 1350.0
st = {}
acc = defaultdict(lambda: defaultdict(float))
k0 = defaultdict(dict)
k1 = defaultdict(dict)
with open(path) as f:
    next(f)
    r = csv.reader(f)
    next(r)
    for row in r:
        row = [x.strip() for x in row]
        run = row[7]
        if run not in runs:
            continue
        c = (int(row[1]), int(row[2]))
        risc = row[3]
        z = row[10]
        ty = row[11]
        t = int(row[5])
        if z.endswith("-KERNEL"):
            if ty == "ZONE_START":
                k0[run][c] = min(k0[run].get(c, t), t)
            else:
                k1[run][c] = max(k1[run].get(c, t), t)
            continue
        key = (run, c, risc, z)
        if ty == "ZONE_START":
            st[key] = t
        elif key in st:
            acc[(run, c)][(risc, z)] += (t - st.pop(key)) / MHZ
for run in sys.argv[2:]:
    t0 = min(k0[run].values())
    ends = sorted(((v - t0) / MHZ, c) for c, v in k1[run].items())[::-1]
    nstall = sum(1 for c in k1[run] if acc[(run, c)][("TRISC_0", "compute_wait_coef")] > 2)
    print(f"run {run}: wall {ends[0][0]:.1f}  cores with coef wait>2us: {nstall}")
    for w, c in ends[:6]:
        a = acc[(run, c)]
        print(
            f"   {c} end {w:6.1f} coef_wait {a[('TRISC_0','compute_wait_coef')]:5.1f} wait_in {a[('TRISC_0','compute_wait_in')]:5.1f} reserve_out {a[('TRISC_2','compute_reserve_out')]:5.1f}"
        )
