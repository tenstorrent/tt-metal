"""compute_drain: per worker core, the last block's drain timeline (us from the replay instance's first kernel
start) in a whole-op trace profile. Usage: python3 drain_from_profile.py <csv> [--top N]"""
import csv, sys, bisect
from collections import defaultdict

F = 1350.0
path = sys.argv[1]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 8
rows = list(csv.reader(open(path)))[2:]
kst = defaultdict(list)
for r in rows:
    if r[10].strip().endswith("-KERNEL") and r[11].strip() == "ZONE_START":
        kst[r[0]].append(int(r[5]))
inst = {}
for c, ts in kst.items():
    ts.sort()
    s = [ts[0]]
    for a, b in zip(ts, ts[1:]):
        if b - a > 5 * F:
            s.append(b)
    inst[c] = s
ev = defaultdict(list)  # (chip,x,y,risc,zone) -> [(type,t)]
for r in rows:
    c = r[0]
    t = int(r[5])
    i = bisect.bisect_right(inst[c], t) - 1
    if i != len(inst[c]) - 1:
        continue
    ev[(c, int(r[1]), int(r[2]), r[3].strip(), r[10].strip())].append((r[11].strip(), (t - inst[c][-1]) / F))


def spans(k):
    out, st = [], None
    for typ, t in sorted(ev.get(k, []), key=lambda a: a[1]):
        if typ == "ZONE_START":
            st = t
        elif st is not None:
            out.append((st, t))
            st = None
    return out


cores = sorted({k[:3] for k in ev if k[4] == "compute_matmul"})
res = []
for c, x, y in cores:
    rd = spans((c, x, y, "NCRISC", "reader_attn_block"))
    ww = spans((c, x, y, "BRISC", "writer_wait_block"))
    lw = spans((c, x, y, "BRISC", "writer_local_write"))
    d = {"rd": rd[-1][1] if rd else None}
    for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
        for z in ("compute_tilize", "compute_matmul"):
            s = spans((c, x, y, risc, z))
            d[f"{z[8:]}{risc[-1]}"] = s[-1] if s else None
    d["ww_end"] = ww[-1][1] if ww else None
    d["lw"] = lw[-1] if lw else None
    end = d["lw"][1] if d["lw"] else d["matmul2"][1]
    res.append((end, (c, x, y), d))
res.sort(reverse=True)
print(
    "core            read_end | tilize u/m/p end | matmul u/m/p start..end | wr_wait_end local_write   drain(end-read)"
)
for end, k, d in res[:top]:
    f = lambda v: f"{v:5.2f}" if v is not None else "  -  "
    print(
        f"{k!s:16s} {f(d['rd'])} | {f(d['tilize0'][1])} {f(d['tilize1'][1])} {f(d['tilize2'][1])} | "
        f"{f(d['matmul0'][0])}..{f(d['matmul0'][1])} {f(d['matmul1'][0])}..{f(d['matmul1'][1])} {f(d['matmul2'][0])}..{f(d['matmul2'][1])} | "
        f"{f(d['ww_end'])} {f(d['lw'][0]) if d['lw'] else '  -  '}..{f(d['lw'][1]) if d['lw'] else '  -  '}  {end - d['rd']:5.2f}"
    )
