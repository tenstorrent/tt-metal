import csv, statistics, sys, glob

d = "/tmp/mhc_r2/whole/"
names = [l.strip().split("::")[-1] for l in open(d + "names.txt")]


def load(f):
    r = [int(x["DEVICE KERNEL DURATION [ns]"]) for x in csv.DictReader(open(f))]
    assert len(r) == len(names), (f, len(r))
    return r


B = [load(d + f"before{i}.csv") for i in (1, 2, 3)]
A = [load(f) for f in sorted(glob.glob(d + "after_b*.csv")) + sorted(glob.glob(d + "x_after_*.csv"))]
print(f"{'case':60s} {'before(med3)':>12s} {'after(med%d)'%len(A):>12s}  delta   before-range  after-range")
for i, n in enumerate(names):
    b = [x[i] / 1000 for x in B]
    a = [x[i] / 1000 for x in A]
    mb, ma = statistics.median(b), statistics.median(a)
    print(
        f"{n:60s} {mb:12.1f} {ma:12.1f} {100*(ma/mb-1):+6.1f}%  {min(b):6.1f}-{max(b):6.1f}  {min(a):6.1f}-{max(a):6.1f}"
    )
