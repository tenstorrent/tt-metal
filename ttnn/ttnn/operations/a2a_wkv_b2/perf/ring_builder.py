# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""a2a_wkv_b2 on one 4-chip ring, matmul first, as an opgraph task graph (pipeline_spec.py refuses a ring).

Every chip of the ring is in the graph (chip=(0, i)). Per chip: all workers read W (resident), then attn one
mt round at a time (mt_block token-tile-rows of every head), tilize + matmul each round, and the 4 fabric_tx
streams (2 ring directions x links) send each finished round's packets (4 bf8 tiles = 4352 B, one scatter write).
Neighbours are 1 hop; the opposite chip's 24 heads go 12 one way, 12 the other (2 hops, fabric routers relay).
`antipode_split=False` sends the whole opposite chunk one way (the unbalanced alternative).
dram_read_tiles in `must` counts attn in 2 KB-equivalents (15,728,640 / 2048 = 7680) plus W's 6144 bf8 tiles
scaled to 2 KB (6,684,672 / 2048 = 3264): 10,944."""

import math

H, MT, KT, NT, TPC = 96, 5, 16, 4, 160
ATTN_ROW = 1024
W_HEAD = KT * NT * 1088
PKT = NT * 1088  # one mt block of one head: 4 tiles


def build(g, p, mt_block=1, cores=96, antipode_split=True):
    hw, c = g.hw, g.cost
    n = hw.axis(1)["size"]
    lo, hi = hw.axis(1)["directions"]  # W, E
    hpc = H // n
    nrounds = MT // mt_block
    chip = lambda i: (0, i)

    def route(src, dst, head):
        k = (dst - src) % n
        if k * 2 < n:
            return hi, k
        if k * 2 > n:
            return lo, n - k
        if antipode_split:
            return (hi if head < hpc // 2 else lo), k
        return hi, k

    lands = {i: [] for i in range(n)}
    firsts = {i: [] for i in range(n)}
    for i in range(n):
        g.task(f"launch_{i}", c.fabric("launch"))
        g.task(f"open_{i}", c.fabric("open"), f"launch_{i}")
        g.task(f"fence_{i}", c.fabric("barrier", n // 2), f"open_{i}")
        g.task(f"first_{i}", c.fabric("first_packet"), f"fence_{i}")
        dram = hw.dram(chip(i))
        prev = [f"launch_{i}"]
        prev = [g.task(f"rdW_{i}", c.dram_stream(H * W_HEAD), prev, on=dram)]
        for rnd in range(nrounds):
            nbytes = H * mt_block * 32 * ATTN_ROW
            rd = g.task(f"rdA_{i}_{rnd}", c.dram_stream(nbytes), prev, on=dram)
            prev = [rd]
            tz = g.task(
                f"tilize_{i}_{rnd}",
                c.compute_llk(
                    "perf_unpack_tilize", math.ceil(H * mt_block * KT / cores), full_ct_dim="8", full_rt_dim="1"
                ),
                rd,
                on=hw.compute(0, chip(i)),
            )
            mm = g.task(
                f"mm_{i}_{rnd}", c.matmul(H * mt_block * NT * KT, "HiFi2", cores=cores), tz, on=hw.compute(0, chip(i))
            )
            # local heads: worker writes its own chip's output L1
            loc = g.task(f"local_{i}_{rnd}", c.noc_stream(hpc * mt_block * PKT), mm, on=hw.store(0, chip(i)))
            lands[i].append(loc)
            # remote heads: per (direction, link) stream, the round's packets
            streams = {}
            for dst in range(n):
                if dst == i:
                    continue
                for hh in range(hpc):
                    d, hops = route(i, dst, hh)
                    link = hh % hw.links
                    streams.setdefault((d, link), []).append((dst, hops))
            for (d, link), items in streams.items():
                for hops in sorted({h for _, h in items}):
                    cnt = sum(1 for _, h in items if h == hops)
                    b = cnt * mt_block * PKT
                    at, pr = i, [mm, f"first_{i}"]
                    step = 1 if d == hi else -1
                    for hp in range(hops):
                        tx = g.task(
                            f"tx_{i}_{rnd}_{d}{link}_{hops}_{hp}",
                            c.link(b),
                            pr,
                            on=hw.link(d, link, chip(at % n)),
                            priority=0 if hp else 1,
                        )
                        if hp == 0:
                            firsts[i].append(tx)
                        at += step
                        pr = [g.task(f"hop_{i}_{rnd}_{d}{link}_{hops}_{hp}", c.fabric("hop"), tx)]
                    lands[at % n].append(g.task(f"land_{i}_{rnd}_{d}{link}_{hops}", c.noc_stream(b), pr))
    for i in range(n):
        g.task(f"exit_{i}", c.fabric("barrier", n // 2), firsts[i])
        g.task(f"close_{i}", c.fabric("close"), [f"exit_{i}"] + lands[i])
