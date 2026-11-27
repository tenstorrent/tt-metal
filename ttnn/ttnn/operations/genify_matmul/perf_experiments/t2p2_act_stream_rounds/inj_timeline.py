"""Per-core zone timeline of the A injectors (cores that have wr_a_send). usage: inj_timeline.py <profile_log_device.csv>"""
import csv, sys, collections

rows = list(csv.reader(open(sys.argv[1])))[2:]
by_run = collections.defaultdict(list)
for r in rows:
    by_run[r[7].strip()].append(r)
R = by_run[max(by_run, key=lambda k: len(by_run[k]))]
t0 = min(int(r[5]) for r in R)
ev = collections.defaultdict(list)
for r in R:
    if r[3].strip() != "BRISC":
        continue
    ev[(int(r[1]), int(r[2]))].append((int(r[5]) - t0, r[10].strip(), r[11].strip()))
inj = [c for c, e in ev.items() if any(z == "wr_a_send" for _, z, _ in e)]
for c in sorted(inj):
    out, open_ = [], {}
    for t, z, typ in sorted(ev[c]):
        if z in ("wr_a_rotate",):
            continue
        if typ == "ZONE_START":
            open_[z] = t
        elif typ == "ZONE_END" and z in open_:
            s = open_.pop(z)
            if z in ("wr_a_recv1",):
                continue
            out.append(f"{z[5:]}@{s/1350:.1f}+{(t-s)/1350:.1f}")
    print(c, " ".join(out))
