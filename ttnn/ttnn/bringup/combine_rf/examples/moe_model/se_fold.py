# Link cost of folding the TP-split shared expert into dispatch (own x rows also to the TP-row chips) and combine
# (SE partial rows home to their owner), vs separate TP all-gather + reduce-scatter. Uniform routing, busiest
# directed (link, plane), rows. GLM-family (routed width = H).
import collections
import random

from global_block import GBPS, STAGES, L, xy_route, yx_route


def run(SP, TP, wtp, wsp, E, K, T, se, seed=0):
    rnd = random.Random(seed)
    N = SP * TP
    epc = E // N
    chips = [(r, c) for r in range(SP) for c in range(TP)]
    o = T // N
    d, cmb = collections.Counter(), collections.Counter()
    for home in chips:
        for t in range(o):
            hosts = [chips[e // epc] for e in rnd.sample(range(E), K)]
            dst = set(hosts) - {home}
            if se:
                dst |= {(home[0], c) for c in range(TP)} - {home}
            best = None
            for f in (yx_route, xy_route):
                links = set()
                for x in dst:
                    links.update(f(home, x, SP, TP, wtp, wsp, (t >> 1) & 1))
                if best is None or len(links) < len(best):
                    best = links
            for h in best:
                d[h + (t % L,)] += 1
            for k, h in enumerate(hosts):
                for hop in yx_route(h, home, SP, TP, wtp, wsp, ((t + k) >> 1) & 1):
                    cmb[hop + ((t + k) % L,)] += 1
            if se:  # the 3 TP neighbours' SE partials of this token come home
                for c in range(TP):
                    src = (home[0], c)
                    if src != home:
                        for hop in yx_route(src, home, SP, TP, wtp, wsp, ((t + c) >> 1) & 1):
                            cmb[hop + ((t + c) % L,)] += 1
    return max(d.values()), max(cmb.values())


def coll(o, TP, wtp):  # separate TP all-gather or reduce-scatter of [TP o, H]: busiest (link, plane) rows
    return (1.5 if wtp else 2.0) * o / L


for name, E, K, H in (("GLM-5.3-Flash", 288, 8, 4096), ("GLM-5.3", 256, 8, 6144), ("DSv4.1-Flash", 384, 6, 5120)):
    for st, T in (("2x4", 5120), ("2x4", 2048), ("4x4", 5120), ("8x4", 5120), ("8x4", 16384)):
        SP, TP, wtp, wsp = STAGES[st]
        if E % (SP * TP):
            continue
        d0, c0 = run(SP, TP, wtp, wsp, E, K, T, False)
        d1, c1 = run(SP, TP, wtp, wsp, E, K, T, True)
        us = lambda r: r * H * 2 / (GBPS * 1e3)
        cc = coll(T // (SP * TP), TP, wtp)
        print(
            f"{name:13s} {st} T{T:6d}: dispatch {us(d0):5.0f} -> {us(d1):5.0f} us (+{us(d1-d0):4.0f}) | combine {us(c0):5.0f} -> {us(c1):5.0f} (+{us(c1-c0):4.0f}) | separate AG / RS floor {us(cc):4.0f} each"
        )
