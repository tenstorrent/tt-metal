# A (fold): 2D drop-off dispatch with the TP-row chips ORed into each own token's mask (the SE's x gather inside dispatch)
# B (user): x all-gather over TP (E/W links only), then 1D column dispatch along SP (N/S only), combine 2D.
# Both: combine 2D with the SE partial rows home to their owners. Busiest directed (link, plane) per axis, us at 25 GB/s.
import collections
import random

from global_block import EFF_AG, EFF_DISP, GBPS, STAGES, L, ring_cover, xy_route, yx_route


def run(SP, TP, wtp, wsp, E, K, T, seed=0):
    rnd = random.Random(seed)
    N = SP * TP
    epc = E // N
    chips = [(r, c) for r in range(SP) for c in range(TP)]
    o = T // N
    dA, d1, cmb = collections.Counter(), collections.Counter(), collections.Counter()
    for home in chips:
        for t in range(o):
            hosts = [chips[e // epc] for e in rnd.sample(range(E), K)]
            dst = (set(hosts) | {(home[0], c) for c in range(TP)}) - {home}
            best = None
            for f in (yx_route, xy_route):
                links = set()
                for x in dst:
                    links.update(f(home, x, SP, TP, wtp, wsp, (t >> 1) & 1))
                if best is None or len(links) < len(best):
                    best = links
            for h in best:
                dA[h + (t % L,)] += 1
            for c in range(TP):  # B: column c's chip of the home row drops the token along its SP ring
                rows = {h[0] for h in hosts if h[1] == c} - {home[0]}
                for (lk,) in ring_cover(SP, home[0], rows, wsp):
                    d1[((lk[0], c), "Y" + lk[1], (t + c) % L)] += 1
            for k, h in enumerate(hosts):
                for hop in yx_route(h, home, SP, TP, wtp, wsp, ((t + k) >> 1) & 1):
                    cmb[hop + ((t + k) % L,)] += 1
            for c in range(TP):
                if (home[0], c) != home:
                    for hop in yx_route((home[0], c), home, SP, TP, wtp, wsp, ((t + c) >> 1) & 1):
                        cmb[hop + ((t + c) % L,)] += 1
    ax = lambda C, a: max((n for k, n in C.items() if k[1][0] == a), default=0)
    return {n: (ax(C, "X"), ax(C, "Y")) for n, C in (("A", dA), ("B", d1), ("c", cmb))}


for name, E, K, Hx, Hr in (
    ("GLM-5.3-Flash", 288, 8, 4096, 4096),
    ("GLM-5.3", 256, 8, 6144, 6144),
    ("DSv4.1-Flash", 384, 6, 5120, 5120),
    ("Kimi K3", 896, 16, 7168, 3584),
):
    for st, T in (("2x4", 5120), ("4x4", 5120), ("8x4", 5120), ("8x4", 16384)):
        SP, TP, wtp, wsp = STAGES[st]
        if E % (SP * TP):
            continue
        r = run(SP, TP, wtp, wsp, E, K, T)
        us = lambda rows, w, eff: rows * w * 2 / (GBPS * 1e3) / eff
        o = T // (SP * TP)
        ag = (1.5 if wtp else 2.0) * o / L * Hx * 2 / (GBPS * 1e3) / EFF_AG  # TP all-gather of x, busiest TP link
        # A: K3's SE x rows are 7168 wide but routed rows 3584: approximate A's dispatch at the routed width
        A = max(us(r["A"][0], Hr, EFF_DISP), us(r["A"][1], Hr, EFF_DISP))
        Bd = us(r["B"][1], Hr, EFF_DISP)
        print(
            f"{name:13s} {st} T{T:6d}: A fold 2D dispatch {A:5.0f} | B AG(TP) {ag:4.0f} + SP dispatch {Bd:5.0f} = serial {ag + Bd:5.0f}, concurrent {max(ag, Bd):5.0f}"
            f" | combine X/Y {us(r['c'][0], Hr, 1):4.0f}/{us(r['c'][1], Hr, 1):4.0f} floor"
        )
