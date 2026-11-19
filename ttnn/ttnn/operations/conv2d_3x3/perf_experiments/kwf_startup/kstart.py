# usage: kstart.py <dev csv[.gz]> <run idx 1-based among runs with >= 8 cores>: per core, each RISC's FW / KERNEL start
# (us from the run's earliest kernel start), sorted by BRISC kernel start; plus the spread per RISC.
import collections, csv, gzip, sys

path, idx = sys.argv[1], int(sys.argv[2])
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    hdr = f.readline()
    freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = csv.reader(f)
    cols = [c.strip() for c in next(rows)]
    ix = {c: i for i, c in enumerate(cols)}
    runs = collections.defaultdict(list)
    for r in rows:
        if len(r) < len(cols):
            continue
        runs[int(r[ix["run host ID"]])].append(
            (
                (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                r[ix["RISC processor type"]].strip(),
                int(r[ix["time[cycles since reset]"]]),
                r[ix["zone name"]].strip(),
                r[ix["type"]].strip(),
            )
        )
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
ev = runs[big[idx - 1]]
st = collections.defaultdict(dict)
en = collections.defaultdict(dict)
for core, risc, t, z, kind in ev:
    if z.endswith("-KERNEL") or z.endswith("-FW"):
        key = (risc, z.split("-")[-1])
        if kind == "ZONE_START":
            st[core][key] = min(st[core].get(key, t), t)
        else:
            en[core][key] = max(en[core].get(key, t), t)
t0 = min(v for c in st.values() for k, v in c.items() if k[1] == "KERNEL")
us = lambda c: (c - t0) / freq
cores = sorted(st, key=lambda c: st[c].get(("BRISC", "KERNEL"), 0))
mode = sys.argv[3] if len(sys.argv) > 3 else "summary"
if mode == "all":
    for c in cores:
        s = st[c]
        print(
            c,
            " ".join(f"{r[0][0]}{r[0][-1]}:{us(s[r]):.2f}" for r in sorted(s) if r[1] == "KERNEL"),
            f"end B {us(en[c].get(('BRISC','KERNEL'), t0)):.2f}",
        )
for risc in ("BRISC", "NCRISC", "TRISC_0"):
    v = sorted(us(st[c][(risc, "KERNEL")]) for c in st if (risc, "KERNEL") in st[c])
    print(
        risc,
        "kernel start q0/q10/q50/q90/q100:",
        " ".join(f"{v[min(len(v)-1, int(q*len(v)))]:.2f}" for q in (0, 0.1, 0.5, 0.9, 1)),
    )
