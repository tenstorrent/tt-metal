"""Per-run (one op dispatch) straggle summary from a profile_log_device.csv.
Usage: runs.py <csv> [labels.jsonl]
Per run: op span (first FW start -> last FW end, us), per-core max / median sums of the DM wait zones,
and the body end (last compute_mix end on TRISC_0, from FW start) max / median over cores."""
import collections, csv, json, sys

path = sys.argv[1]
labels = [json.loads(l) for l in open(sys.argv[2])] if len(sys.argv) > 2 else []
with open(path) as fh:
    freq = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(fh, skipinitialspace=True))
us = lambda c: c / freq  # noqa: E731
ZON = [
    ("reader_barrier", "NCRISC"),
    ("reader_w_mcast", "NCRISC"),
    ("reader_in_reserve", "NCRISC"),
    ("compute_in_wait", "TRISC_0"),
    ("writer_barrier", "BRISC"),
]
st = {}
t0 = {}
t1 = collections.defaultdict(int)
dur = collections.defaultdict(lambda: collections.defaultdict(float))
body = collections.defaultdict(dict)
for r in rows:
    run = r["run host ID"]
    k = (run, r["core_x"], r["core_y"], r["RISC processor type"], r["zone name"])
    t = int(r["time[cycles since reset]"])
    z = r["zone name"]
    if r["type"] == "ZONE_START":
        st[k] = t
        if z.endswith("-FW"):
            t0[run] = min(t0.get(run, t), t)
    elif r["type"] == "ZONE_END" and k in st:
        s = st.pop(k)
        if z.endswith("-FW"):
            t1[run] = max(t1[run], t)
        if (z, k[3]) in ZON:
            dur[(run, z)][(k[1], k[2])] += t - s
        if z == "compute_mix" and k[3] == "TRISC_0":
            body[run][(k[1], k[2])] = t
runs = sorted(t0, key=int)
hdr = (
    "run  label                             span  "
    + "  ".join(f"{z[:14]:>14s}(max/med)" for z, _ in ZON)
    + "   body_end(max/med)"
)
print(hdr)
lab = labels[-len(runs) :] if labels else [{}] * len(runs)
for i, run in enumerate(runs):
    L = lab[i] if i < len(lab) else {}
    parts = []
    for z, _ in ZON:
        v = sorted(dur[(run, z)].values()) or [0]
        parts.append(f"{us(v[-1]):9.1f}/{us(v[len(v)//2]):6.1f}      ")
    b = sorted(body[run].values()) or [t0[run]]
    print(
        f"{run:>4s} {L.get('case','')[:22]:22s} {L.get('who',''):7s} {us(t1[run]-t0[run]):6.1f}  "
        + " ".join(parts)
        + f"  {us(b[-1]-t0[run]):6.1f}/{us(b[len(b)//2]-t0[run]):6.1f}"
    )
