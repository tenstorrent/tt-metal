"""For one run: cores sorted by body end (last TRISC_0 compute_mix end), with their DM zone sums. Usage: late.py csv run [N]"""
import collections, csv, sys

path, run = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 12
with open(path) as fh:
    freq = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = [r for r in csv.DictReader(fh, skipinitialspace=True) if r["run host ID"] == run]
Z = ["reader_barrier", "reader_w_mcast", "reader_in_reserve", "compute_in_wait", "writer_barrier", "writer_out"]
st = {}
t0 = None
d = collections.defaultdict(float)
be = {}
first_mix = {}
for r in rows:
    k = (r["core_x"], r["core_y"], r["RISC processor type"], r["zone name"])
    t = int(r["time[cycles since reset]"])
    if r["type"] == "ZONE_START":
        st[k] = t
        if r["zone name"].endswith("-FW"):
            t0 = t if t0 is None else min(t0, t)
        if r["zone name"] == "compute_mix" and k[2] == "TRISC_0":
            first_mix.setdefault(k[:2], t)
    elif r["type"] == "ZONE_END" and k in st:
        s = st.pop(k)
        d[(k[0], k[1], k[3])] += t - s
        if r["zone name"] == "compute_mix" and k[2] == "TRISC_0":
            be[k[:2]] = t
cores = sorted(be, key=lambda c: -be[c])
print("core     body_end mix0_start " + " ".join(f"{z[:13]:>13s}" for z in Z))
for c in cores[:N] + [None] + cores[-3:]:
    if c is None:
        print("...")
        continue
    print(
        f"{c[0]:>3s},{c[1]:<4s} {(be[c]-t0)/freq:8.1f} {(first_mix[c]-t0)/freq:8.1f}  "
        + " ".join(f"{d[(c[0],c[1],z)]/freq:13.1f}" for z in Z)
    )
