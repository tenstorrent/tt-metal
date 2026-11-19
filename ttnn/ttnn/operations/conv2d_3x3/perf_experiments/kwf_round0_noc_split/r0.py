# usage: r0.py <device csv(.gz)> [run_index ...]: per run (1-based among runs with >= 8 cores), the round-0 chain on the
# receivers, each time on the receiver's own clock (us from the run's earliest kernel start):
#   kstart   BRISC kernel start
#   r0flag   end of the writer's first writer_weight_fetch (round 0's flag seen)
#   wfill    end of the writer's last writer_fill that STARTED before r0flag (its NoC1 first-block fill issue)
#   rfill    end of the reader's first reader_fill (the reader's half landed and every batch pushed)
#   c.r0     end of compute's first compute_kwf_weight_wait (TRISC_0): round 0 in compute's hands
#   c.end    TRISC_2 kernel end
# p50 / max over receivers, plus the servers' round-0 send window (first server_send start .. first server_flush end).
import collections, csv, gzip, statistics as st, sys

path = sys.argv[1]
opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt") as f:
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
want = [int(v) for v in sys.argv[2:]] or list(range(1, len(big) + 1))
for idx in want:
    ev = sorted(runs[big[idx - 1]], key=lambda e: e[2])
    t0 = min(e[2] for e in ev if e[3].endswith("-KERNEL") and e[4] == "ZONE_START")
    us = lambda c: (c - t0) / freq
    z = collections.defaultdict(list)
    stack = collections.defaultdict(list)
    for core, risc, t, name, kind in ev:
        if kind == "ZONE_START":
            stack[(core, risc, name)].append(t)
        elif stack[(core, risc, name)]:
            s = stack[(core, risc, name)].pop()
            z[(core, risc, name)].append((us(s), us(t)))
    kend = max(us(e[2]) for e in ev if e[3].endswith("-KERNEL") and e[4] == "ZONE_END")
    servers = sorted({k[0] for k in z if k[2].startswith("server_")})
    recv = sorted({k[0] for k in z if k[2] == "writer_weight_fetch"} - set(servers))
    out = []
    for c in recv:
        g = lambda r, n: sorted(z.get((c, r, n), []))
        ks = g("BRISC", "BRISC-KERNEL")
        wf = g("BRISC", "writer_weight_fetch")
        r0 = wf[0][1] if wf else -1
        fl = [e for e in g("BRISC", "writer_fill") if e[0] < r0]
        rf = g("NCRISC", "reader_fill")
        ww = g("TRISC_0", "compute_kwf_weight_wait")
        ck = g("TRISC_2", "TRISC-KERNEL")
        out.append(
            (
                c,
                ks[0][0] if ks else -1,
                r0,
                fl[-1][1] if fl else -1,
                rf[0][1] if rf else -1,
                ww[0][1] if ww else -1,
                ck[0][1] if ck else -1,
            )
        )
    names = ["kstart", "r0flag", "wfill", "rfill", "c.r0", "c.end"]
    line = f"run {idx} ({big[idx-1]}): {len(recv)} recv, end {kend:5.2f} |"
    for i, n in enumerate(names):
        v = [r[i + 1] for r in out if r[i + 1] >= 0]
        if v:
            line += f" {n} {st.median(v):5.2f}/{max(v):5.2f}"
    print(line)
    for s in servers:
        g = lambda n: sorted(z.get((s, "BRISC", n), []))
        snd, fls, ack = g("server_send"), g("server_flush"), g("server_wait_ack")
        rd = sorted(z.get((s, "NCRISC", "server_read"), []))
        print(
            f"   server {s}: ack_end {ack[0][1] if ack else -1:5.2f} read0 {rd[0][0] if rd else -1:5.2f}-{rd[0][1] if rd else -1:5.2f}"
            f" send0 {snd[0][0] if snd else -1:5.2f} flush0_end {fls[0][1] if fls else -1:5.2f}"
        )
