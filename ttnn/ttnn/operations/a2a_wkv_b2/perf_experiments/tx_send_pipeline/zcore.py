"""Mean duration (us) per zone per tx-core position (x,y) over all chips and replays 1.., for zones matching argv[2:].
Usage: python3 zcore.py <csv[.gz]> zone_prefix..."""
import csv, sys, gzip
from collections import defaultdict

F = 1350.0
op = gzip.open if sys.argv[1].endswith(".gz") else open
rows = list(csv.reader(op(sys.argv[1], "rt")))[2:]
pref = sys.argv[2:]
st = {}
d = defaultdict(list)
for r in rows:
    z = r[10].strip()
    if not any(z.startswith(p) for p in pref):
        continue
    k = (r[0], r[1], r[2], r[3].strip(), z)
    if r[11].strip() == "ZONE_START":
        st[k] = int(r[5])
    elif k in st:
        d[(z, (int(r[1]), int(r[2])))].append((int(r[5]) - st.pop(k)) / F)
for (z, xy), v in sorted(d.items()):
    v2 = v[8:] if len(v) > 8 else v  # drop replay 0 (one per chip)
    print(f"{z:24s} {xy}  n={len(v2):3d} mean={sum(v2)/len(v2):6.2f} max={max(v2):6.2f}")
