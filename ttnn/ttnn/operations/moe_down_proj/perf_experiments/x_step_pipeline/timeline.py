"""Per-step x chain timeline from the reader's x_send / x_receive / x_forward / x_reserve zones.

Usage: python timeline.py profile_log_device.csv [row_y]
Per rect row (cores sharing core_y): for each step, the sender's send zone vs its receivers' receive zones.
Columns (us): per = last receiver data(s) - data(s-1); gapR = slowest receiver's own data(s-1) -> ack(s) gap
(ack = receive start); lateS = sender send start - last ack; xfer = last data(s) - max(send start, last ack);
sdur = send zone duration; fwd = forwarder x_forward(s) duration.
"""
import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
CH = "w" if "--w" in sys.argv else "x"
sys.argv = [a for a in sys.argv if a != "--w"]
RISC = "BRISC" if CH == "w" else "NCRISC"
AX = 0 if CH == "w" else 1  # w lines are rect columns (same core_x), x groups are rect rows (same core_y)
with open(path) as f:
    first = f.readline()
    freq = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(f, skipinitialspace=True))
run_id = max(int(r["run host ID"]) for r in rows)
ZONES = tuple(f"{CH}_{z}" for z in ("send", "receive", "forward", "reserve"))
open_ = {}
ev = defaultdict(list)  # core -> [(zone, start, end)]
for r in rows:
    if int(r["run host ID"]) != run_id or r["RISC processor type"] != RISC or r["zone name"] not in ZONES:
        continue
    core = (int(r["core_x"]), int(r["core_y"]))
    key = (core, r["zone name"])
    t = int(r["time[cycles since reset]"])
    if r["type"] == "ZONE_START":
        open_[key] = t
    elif r["type"] == "ZONE_END" and key in open_:
        ev[core].append((r["zone name"], open_.pop(key), t))
us = lambda c: c / freq
t0 = min(s for core in ev for _, s, _ in ev[core])
ys = sorted({c[1] for c in ev})
strip_y = max(ys)
rect_ys = [y for y in ys if y != strip_y]
if CH == "w":
    rect_ys = sorted({c[0] for c in ev if c[1] != strip_y})
sel = [int(sys.argv[2])] if len(sys.argv) > 2 else rect_ys
summary = defaultdict(list)
for y in sel:
    cores = sorted(c for c in ev if c[AX] == y and (CH == "x" or c[1] != strip_y))
    steps = {}
    fwd = {}
    for c in cores:
        xs = [e for e in ev[c] if e[0] in (f"{CH}_send", f"{CH}_receive")]
        fw = [e for e in ev[c] if e[0] == f"{CH}_forward"]
        for i, e in enumerate(sorted(xs, key=lambda e: e[1])):
            steps.setdefault(i, []).append((c, e))
        for i, e in enumerate(sorted(fw, key=lambda e: e[1])):
            fwd[i] = e
    prev_end = {}
    prev_last = None
    if len(sel) == 1:
        print(f"row core_y={y}: {len(cores)} cores")
        print(" s  sender  per   gapR  gapS  lateS  xfer  sdur  fwd   t_data")
    for s in sorted(steps):
        evs = steps[s]
        snd = [(c, e) for c, e in evs if e[0] == f"{CH}_send"]
        rcv = [(c, e) for c, e in evs if e[0] == f"{CH}_receive"]
        if not snd or not rcv:
            continue
        sc, se = snd[0]
        last_ack_c, last_ack = max(((c, e[1]) for c, e in rcv), key=lambda v: v[1])
        data_last = max(e[2] for _, e in rcv)
        gap = last_ack - prev_end[last_ack_c] if last_ack_c in prev_end else float("nan")
        per = data_last - prev_last if prev_last is not None else float("nan")
        late = se[1] - last_ack
        xfer = data_last - max(se[1], last_ack)
        f = fwd.get(s)
        fd = (f[2] - f[1]) if f else 0
        gaps = se[1] - prev_end[sc] if sc in prev_end else float("nan")
        vals = dict(
            per=us(per), gapR=us(gap), gapS=us(gaps), lateS=us(late), xfer=us(xfer), sdur=us(se[2] - se[1]), fwd=us(fd)
        )
        if s > 0:
            for k, v in vals.items():
                summary[k].append(v)
        if len(sel) == 1:
            print(
                f"{s:2d} {sc[0]:2d},{sc[1]:2d} {vals['per']:5.2f} {vals['gapR']:5.2f} {vals['gapS']:5.2f} {vals['lateS']:6.2f} "
                f"{vals['xfer']:5.2f} {vals['sdur']:5.2f} {vals['fwd']:5.2f} {us(data_last - t0):7.2f}  lastack={last_ack_c}"
            )
        for c, e in evs:
            prev_end[c] = e[2]
        prev_last = data_last
print("median over rows/steps (us):", {k: round(statistics.median(v), 2) for k, v in summary.items()})
print("mean   over rows/steps (us):", {k: round(statistics.mean(v), 2) for k, v in summary.items()})
