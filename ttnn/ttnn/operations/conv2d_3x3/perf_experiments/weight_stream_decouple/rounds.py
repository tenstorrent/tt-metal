# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: per-round weight-stream timeline of one program run from the device profiler log.

    python3 rounds.py <dev.csv[.gz]> <run ordinal among >=100-core runs> [--rounds N]

Sender (the BRISC with wsd_read zones, or writer_weight_send): per round reserve wait, read, ack wait, send.
Receivers: per round the reserve wait and the fetch (ack + data wait) per core, the time from the sender's send end to
the slowest / median receiver's fetch end, and each round's spread of reserve ends (when each core's compute freed the
slot = when it acks). All times us.
"""

import collections
import csv
import gzip
import statistics
import sys


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        header = f.readline()
        freq = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
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
    return freq, runs


def zones(events):
    """(core, risc) -> zone -> [(start, end)] in order."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    stacks = collections.defaultdict(list)
    for core, risc, t, z, kind in sorted(events, key=lambda e: e[2]):
        if kind == "ZONE_START":
            stacks[(core, risc, z)].append(t)
        elif kind == "ZONE_END" and stacks[(core, risc, z)]:
            out[(core, risc)][z].append((stacks[(core, risc, z)].pop(), t))
    return out


def main():
    path, ordinal = sys.argv[1], int(sys.argv[2])
    nr = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 40
    freq, runs = load(path)
    big = [rid for rid in sorted(runs) if len({e[0] for e in runs[rid]}) >= 100]
    rid = big[ordinal]
    zs = zones(runs[rid])
    us = lambda c: c / freq
    t0 = min(e[2] for e in runs[rid])
    kern = {k: v["BRISC-KERNEL"][0] for k, v in zs.items() if "BRISC-KERNEL" in v}
    print(f"run {rid}: {len(kern)} writers, kernel span max {us(max(e - s for s, e in kern.values())):.1f} us")
    senders = [k for k, v in zs.items() if k[1] == "BRISC" and ("wsd_read" in v or "writer_weight_send" in v)]
    if not senders:
        print("no sender zones")
        return
    snd = senders[0]
    S = zs[snd]
    sends = S.get("wsd_send") or S.get("writer_weight_send")
    reads = S.get("wsd_read", [])
    acks = S.get("wsd_ackwait", [])
    res = S.get("writer_weight_reserve", [])
    print(f"sender {snd}: {len(sends)} sends zoned, {len(res)} reserves")
    recv = [k for k, v in zs.items() if k[1] == "BRISC" and k != snd and "writer_weight_fetch" in v]
    # Receiver fetch zones per round (index = round, from the first non-slab-direct round on).
    print(
        "round | snd reserve | read | ackwait | send | send_end(t) | recv fetch p50/max | land lag p50/max"
        " | ack spread (reserve end max-min) | recv reserve p50/max"
    )
    for r in range(min(nr, len(sends))):
        rs = res[r] if r < len(res) else (0, 0)
        rd = reads[r] if r < len(reads) else (0, 0)
        ak = acks[r] if r < len(acks) else (0, 0)
        sd = sends[r]
        fetch = [zs[k]["writer_weight_fetch"][r] for k in recv if len(zs[k]["writer_weight_fetch"]) > r]
        rres = [zs[k]["writer_weight_reserve"][r] for k in recv if len(zs[k].get("writer_weight_reserve", [])) > r]
        fd = [us(e - s) for s, e in fetch]
        lag = [us(e - sd[1]) for s, e in fetch]
        rend = [e for s, e in rres]
        rdur = [us(e - s) for s, e in rres]
        print(
            f"{r:3d} | {us(rs[1]-rs[0]):7.2f} | {us(rd[1]-rd[0]):5.2f} | {us(ak[1]-ak[0]):6.2f} | {us(sd[1]-sd[0]):5.2f} |"
            f" {us(sd[1]-t0):8.1f} | {statistics.median(fd) if fd else 0:6.2f} / {max(fd) if fd else 0:6.2f} |"
            f" {statistics.median(lag) if lag else 0:6.2f} / {max(lag) if lag else 0:6.2f} |"
            f" {us(max(rend)-min(rend)) if rend else 0:6.2f} | {statistics.median(rdur) if rdur else 0:6.2f} / {max(rdur) if rdur else 0:6.2f}"
        )
    # Period of the sender's sends.
    ends = [e for s, e in sends]
    per = [us(b - a) for a, b in zip(ends, ends[1:])]
    if per:
        print(f"send period p50 {statistics.median(per):.2f} us, mean {statistics.mean(per):.2f}, n {len(per)}")
    tot = lambda zl: sum(us(e - s) for s, e in zl)
    print(
        f"sender sums over zoned rounds: reserve {tot(res):.1f} read {tot(reads):.1f} ackwait {tot(acks):.1f} send {tot(sends):.1f}"
    )
    # Per-core compute-side weight waits are not zoned; give receivers' fetch sums.
    fs = sorted((tot(zs[k]["writer_weight_fetch"]), k) for k in recv)
    print(
        f"receiver fetch sums: min {fs[0][0]:.1f} p50 {fs[len(fs)//2][0]:.1f} max {fs[-1][0]:.1f} (n zoned {len(zs[recv[0]]['writer_weight_fetch'])})"
    )


if __name__ == "__main__":
    main()
