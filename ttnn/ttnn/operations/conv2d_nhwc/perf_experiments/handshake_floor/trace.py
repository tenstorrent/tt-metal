"""Per-core act (hf_a) / weight (hf_w) landing timestamps from an HF_EVTRACE run: python3 -I trace.py <csv>"""
import csv, sys, collections

rows = list(csv.reader(open(sys.argv[1])))[1:]
h = [c.strip() for c in rows[0]]
ix = {c: i for i, c in enumerate(h)}
progs = collections.Counter(r[ix["run host ID"]].strip() for r in rows[1:] if len(r) >= len(h))
print("programs:", dict(progs))
ev = collections.defaultdict(list)
for r in rows[1:]:
    if len(r) < len(h):
        continue
    n = r[ix["zone name"]].strip()
    if n in ("hf_a", "hf_w"):
        ev[(int(r[ix["core_x"]]), int(r[ix["core_y"]]))].append(
            (int(r[ix["time[cycles since reset]"]]), n, int(r[ix["data"]]))
        )
for core in sorted(ev)[:: max(1, len(ev) // 6)]:
    e = sorted(ev[core])
    a = [(t, d) for t, n, d in e if n == "hf_a"]
    w = [(t, d) for t, n, d in e if n == "hf_w"]
    da = [a[i + 1][0] - a[i][0] for i in range(len(a) - 1)]
    dw = [w[i + 1][0] - w[i][0] for i in range(len(w) - 1)]
    print(core, "act gaps", da[:12], "mean", sum(da) // max(1, len(da)))
    print(
        core,
        "w   gaps",
        dw[:12],
        "mean",
        sum(dw) // max(1, len(dw)),
        "own w:",
        [d & 0xFFFF for t, d in w if d >> 16][:4],
    )
    t0 = e[0][0]
    print("   seq:", [(t - t0, n[-1], d & 0xFFFF, d >> 16) for t, n, d in e[:24]])
