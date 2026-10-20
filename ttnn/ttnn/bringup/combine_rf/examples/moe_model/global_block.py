# MoE block layouts on Galaxy stages with MEASURED efficiencies (2026-10-20).
# Layouts (per chip, SP x TP stage, chunk S, q = S / SP row tokens, o = S / N own tokens):
#   2D-rep : token-split residual, replicated shared expert (SE) on own tokens, 2D dispatch / combine (built today)
#   2D-tp  : token-split residual, TP-split SE (x all-gather over TP + SE on q rows + reduce-scatter by tokens)
#   own2D  : H-split residual (main's), x all-gather over TP first, declared owners -> 2D dispatch, combine lands
#            H/TP slices on every chip of the home row (no routed reduce-scatter), SE TP-split
#   1D     : token-split at the edges: x all-gather over TP, column dispatch along SP (drop-off), combine along SP
#            into fp32 partials of the row's q tokens, ONE reduce-scatter over TP (routed + SE partials)
# Link loads from simulated uniform routing; efficiencies from the LoudBox measurements (worklog refs inline).
import collections
import math
import random
import sys

GBPS, L = 25.0, 2  # per link per direction, links per direction
DRAM = 512e9
# measured efficiencies
EFF_DISP = 0.82  # sparse 2D dispatch vs its busiest-link floor (K3 LB, 2026-10-16)
EFF_AG, EFF_RS = 0.87, 0.74  # TP4 ttnn all-gather / reduce-scatter vs link floor (zz_scratch_k3_se_tp, 2026-10-17)
SE_GBPS = 338e9  # lin2d SE weight stream (K3: 140 MB in 270 + 144 us)
SE_TFLOPS = 245e12  # dense / lin2d SE (K3 gate linear at M 1280; lin2d SE at 320 rows >= 204)
LIN_GBPS = 340e9  # lin2d latent projections
FUSED_OVH = 0.05  # fused flat + combine: max(f, c) + 5% of the smaller (GLM-Flash S640 uniform: 1305 / 1113 -> 1362)
W_EFF, TFLOPS, A_PER_H = 0.98, 407e12, 0.062 / 3584  # flat model (linkbound3, calibrated on K3 / V4.1 single chip)

M = {  # E, K, H (residual), Hr (routed width on the wire), I, Is (shared), eff_c (2D combine vs link floor, full-L1 arena)
    "Kimi K3": dict(E=896, K=16, H=7168, Hr=3584, I=3072, Is=6144, eff_c=0.81, latent=True),
    "GLM-5.3": dict(E=256, K=8, H=6144, Hr=6144, I=2048, Is=2048, eff_c=0.75),
    "GLM-5.3-Flash": dict(E=288, K=8, H=4096, Hr=4096, I=2048, Is=2048, eff_c=0.75),
    "DSv4.1-Flash": dict(E=384, K=6, H=5120, Hr=5120, I=2304, Is=2304, eff_c=0.70),
    "Kimi K2.7": dict(E=384, K=8, H=7168, Hr=7168, I=2048, Is=2048, eff_c=0.75),
}
STAGES = {
    "4x4": (4, 4, True, False),
    "8x4": (8, 4, True, True),
    "2x4": (2, 4, False, False),
}  # SP, TP, wrap TP, wrap SP


def step(n, a, b, wrap, tie_plus):
    if wrap:
        f = (b - a) % n
        plus = f * 2 < n or (f * 2 == n and tie_plus)
        return (1 if plus else -1), (f if plus else n - f)
    return (1 if b > a else -1), abs(b - a)


def yx_route(src, dst, SP, TP, wtp, wsp, tie):
    hops, (r, c) = [], src
    s, k = step(SP, r, dst[0], wsp, tie)
    for _ in range(k):
        hops.append(((r, c), "Y+" if s > 0 else "Y-"))
        r = (r + s) % SP
    s, k = step(TP, c, dst[1], wtp, tie)
    for _ in range(k):
        hops.append(((r, c), "X+" if s > 0 else "X-"))
        c = (c + s) % TP
    return hops


def ring_cover(n, src, dests, wrap):
    """Links (as (pos, dir)) of the cheapest drop-off covering dests from src on a ring / line of n."""
    if not dests:
        return []
    if not wrap:
        up = max((d for d in dests if d > src), default=src)
        dn = min((d for d in dests if d < src), default=src)
        return [((p, "+"),) for p in range(src, up)] + [((p, "-"),) for p in range(src, dn, -1)]
    off = sorted((d - src) % n for d in dests)
    best = None
    for i in range(len(off) + 1):  # first i go +, rest go -
        a = off[i - 1] if i else 0
        b = n - off[i] if i < len(off) else 0
        if best is None or a + b < best[0]:
            best = (a + b, a, b)
    _, a, b = best
    return [(((src + j) % n, "+"),) for j in range(a)] + [(((src - j) % n, "-"),) for j in range(b)]


def sim(SP, TP, wtp, wsp, E, K, T, tie_split=False, seed=0):
    """Busiest-link rows (per directed link and link plane) for: 2D combine (one row per slot), 2D dispatch
    (drop-off trees, X- or Y-first by fewer links), 1D column dispatch (drop-off on the SP ring), 1D combine."""
    rnd = random.Random(seed)
    N = SP * TP
    epc = E // N
    chips = [(r, c) for r in range(SP) for c in range(TP)]
    S_own = T // N
    c2, d2, d1, c1 = (collections.Counter() for _ in range(4))
    for hi, home in enumerate(chips):
        for t in range(S_own):
            ex = rnd.sample(range(E), K)
            hosts = [chips[e // epc] for e in ex]
            # 2D combine: every slot's row home
            for k, h in enumerate(hosts):
                tie = ((t + k) >> 1) & 1 if tie_split else True
                for hop in yx_route(h, home, SP, TP, wtp, wsp, tie):
                    c2[hop + ((t + k) % L,)] += 1
            # 2D dispatch: tree from home to the set of host chips
            dst = set(hosts) - {home}
            best = None
            for order in ("YX", "XY"):
                links = set()
                for d in dst:
                    if order == "YX":
                        rt = yx_route(home, d, SP, TP, wtp, wsp, ((t >> 1) & 1) if tie_split else True)
                    else:
                        rt = xy_route(home, d, SP, TP, wtp, wsp, ((t >> 1) & 1) if tie_split else True)
                    links.update(rt)
                if best is None or len(links) < len(best):
                    best = links
            for hop in best:
                d2[hop + (t % L,)] += 1
            # 1D: token t of home row r reaches column c's SP ring from chip (r, c); each column drops off its rows
            r0 = home[0]
            for c in range(TP):
                rows = {h[0] for h in hosts if h[1] == c} - {r0}
                for (lk,) in ring_cover(SP, r0, rows, wsp):
                    d1[((lk[0], c), "Y" + lk[1], (t + c) % L)] += 1
            # 1D combine: each slot's row back along its column to (r0, c)
            for k, h in enumerate(hosts):
                sgn, n = step(SP, h[0], r0, wsp, (((t + k) >> 1) & 1) if tie_split else False)
                r = h[0]
                for _ in range(n):
                    c1[((r, h[1]), "Y+" if sgn > 0 else "Y-", (t + k) % L)] += 1
                    r = (r + sgn) % SP
    res = {k: max(v.values()) if v else 0 for k, v in (("c2", c2), ("d2", d2), ("d1", d1), ("c1", c1))}
    for k, v in (("c2", c2), ("d2", d2), ("d1", d1), ("c1", c1)):
        for ax in "XY":
            res[k + ax] = max((n for key, n in v.items() if key[1][0] == ax), default=0)
    # dispatch + combine on the same directed link (the full-overlap middle carries both)
    for a, b in (("d2", "c2"), ("d1", "c1")):
        both = collections.Counter()
        both.update(locals()[a])
        both.update(locals()[b])
        for ax in "XY":
            res[a + b + ax] = max((n for key, n in both.items() if key[1][0] == ax), default=0)
    return res


def xy_route(src, dst, SP, TP, wtp, wsp, tie):
    hops, (r, c) = [], src
    s, k = step(TP, c, dst[1], wtp, tie)
    for _ in range(k):
        hops.append(((r, c), "X+" if s > 0 else "X-"))
        c = (c + s) % TP
    s, k = step(SP, r, dst[0], wsp, tie)
    for _ in range(k):
        hops.append(((r, c), "Y+" if s > 0 else "Y-"))
        r = (r + s) % SP
    return hops


def us_link(rows, width, eff, bpe=2):
    return rows * width * bpe / (GBPS * 1e3) / eff


def tp_coll(rows, width, TP, wtp, eff, bpe=2):
    """All-gather / reduce-scatter of [rows, width] over TP: busiest link 1.5 / TP shards (ring), 2 / TP (line)."""
    sh = (1.5 if wtp else 2.0) / TP
    return sh * rows * width * bpe / (2 * GBPS * 1e3) / eff  # 2 links per direction


def flat_us(p, N, T):
    epc = p["E"] // N
    me = T * p["K"] / p["E"]
    w0 = 3 * p["Hr"] * p["I"] * 0.5625 / (DRAM * W_EFF) * 1e6
    dram, comp = w0 + A_PER_H * p["Hr"] * me, 6 * p["Hr"] * p["I"] / TFLOPS * 1e6 * me
    return epc * max(dram, comp)


def se_us(p, rows, frac):
    w = 3 * p["H"] * p["Is"] * 1.0625 * frac
    fl = 6 * rows * p["H"] * p["Is"] * frac
    return max(w / SE_GBPS, fl / SE_TFLOPS) * 1e6, w


def lat_us(p, rows):  # K3 latent down + up, replicated weights, on `rows`
    if not p.get("latent"):
        return 0.0
    w = 2 * p["H"] * p["Hr"] * 1.0625
    fl = 2 * 2 * rows * p["H"] * p["Hr"]
    return max(w / LIN_GBPS, fl / SE_TFLOPS) * 1e6


def small(p, N, T):
    """Small serial ops (measured on the LB K3 trace, scaled): router 86, routing gather 36 (x N / 8 hops ~ log),
    planner 57 at 41k entries (linear in S K), dispatch prep + zero 19, norm etc 24, final sum 30."""
    return dict(router=86, gather=36 * (1 + math.log2(N / 8) * 0.5), planner=57 * T * p["K"] / 40960, prep=19, misc=54)


def block(p, st, T, B, levers=()):
    SP, TP, wtp, wsp = STAGES[st]
    N = SP * TP
    q = T // SP
    o = T // N
    eff_c = p["eff_c"]
    Hr = p["Hr"]
    wire = 1.0 if "fp8" not in levers else 0.53
    f = flat_us(p, N, T)
    sm = small(p, N, T)
    smt = sum(sm.values())
    out = {}
    # ---- 2D-rep (built) ----
    c2 = us_link(B["c2"], Hr, eff_c) * wire
    d2 = us_link(B["d2"], Hr, EFF_DISP) * wire
    fused = max(f, c2) + FUSED_OVH * min(f, c2)
    se_rep, se_rep_w = se_us(p, o, 1.0)
    lat2d = lat_us(p, o)
    pre = sm["router"] + sm["gather"] + sm["planner"] + sm["prep"] + lat2d / 2
    post = lat2d / 2 + sm["misc"]
    serial = pre + d2 + fused + se_rep + post
    # overlapped: SE under gather + planner + dispatch (links / latency), rest of it in the middle as extra DRAM
    win = pre + d2  # the SE needs no routing: it can start right after the norm
    ov = pre + d2 + fused + max(0.0, se_rep - win) + post
    out["2D-rep"] = dict(serial=serial, ov=ov, f=f, c=c2, d=d2, se=se_rep, edge=pre + post, coll=0)
    # ---- 2D-tp ----
    ag = tp_coll(q, p["H"], TP, wtp, EFF_AG)
    rs = tp_coll(q, p["H"], TP, wtp, EFF_RS)
    se_tp, _ = se_us(p, q, 1.0 / TP)
    serial = pre + d2 + fused + ag + se_tp + rs + post
    # overlapped: AG + SE under gather / planner / dispatch, RS in the middle (links; the combine's TP leg shares them)
    ov = pre + d2 + max(fused, 0) + max(0, ag + se_tp - win) + post
    out["2D-tp"] = dict(serial=serial, ov=ov, f=f, c=c2, d=d2, se=se_tp, edge=pre + post, coll=ag + rs)
    # ---- own2D: x AG on the critical path, 2D dispatch / combine (H slices home: same busiest link), SE TP ----
    lat_q = lat_us(p, q)  # latent down on the gathered rows of every chip? owners only: o rows
    pre_o = sm["router"] + sm["gather"] + sm["planner"] + sm["prep"] + lat2d / 2
    serial = ag + pre_o + d2 + fused + se_tp + rs + post
    ov = (
        max(ag, sm["router"] + sm["gather"] + sm["planner"])
        + sm["prep"]
        + lat2d / 2
        + d2
        + fused
        + post
        + max(0, se_tp - d2)
    )
    out["own2D"] = dict(serial=serial, ov=ov, f=f, c=c2, d=d2, se=se_tp, edge=0, coll=ag + rs)
    # ---- 1D ----
    c1 = us_link(B["c1"], Hr, eff_c) * wire
    d1 = us_link(B["d1"], Hr, EFF_DISP) * wire
    fused1 = max(f, c1) + FUSED_OVH * min(f, c1)
    # K3: latent down on q rows (every column chip dispatches its picks of all row tokens); latent partials
    # reduce-scattered in latent space (half x-sized), latent up on own tokens; SE RS separate (overlapped)
    if p.get("latent"):
        rs_routed = tp_coll(q, Hr, TP, wtp, EFF_RS)
        pre1 = sm["router"] + sm["gather"] + sm["planner"] * 0.25 * TP / TP + sm["prep"] + lat_us(p, q) / 2
        post1 = rs_routed + lat2d / 2 + sm["misc"]
    else:
        rs_routed = rs
        pre1 = sm["router"] + sm["gather"] + sm["planner"] + sm["prep"]
        post1 = rs + sm["misc"]
    serial = ag + pre1 + d1 + fused1 + se_tp + post1
    ov = (
        max(ag, sm["router"] + sm["gather"] + sm["planner"])
        + (pre1 - sm["router"] - sm["gather"] - sm["planner"])
        + d1
        + fused1
        + post1
        + max(0, se_tp - d1)
    )
    out["1D"] = dict(serial=serial, ov=ov, f=f, c=c1, d=d1, se=se_tp, edge=0, coll=ag + rs_routed)
    # ---- full overlap (weights first: dispatch, combine, SE under the expert weight stream) ----
    # middle = max(DRAM: flat + SE share + latent weights at the flat's efficiency, busiest link: dispatch + combine
    # (+ the SE's TP collectives)); edges = router / gather / planner (+ x AG where it gates dispatch) and the tail
    deff = DRAM * 0.88
    lat_w = (2 * p["H"] * p["Hr"] * 1.0625) if p.get("latent") else 0
    se_w = 3 * p["H"] * p["Is"] * 1.0625
    head = sm["router"] + sm["gather"] + sm["planner"] + sm["prep"]

    def lk(
        key, ax, extra=0.0
    ):  # dispatch rows at the dispatch efficiency, combine rows at the combine's: use the worse
        return us_link(B[key + ax], Hr, min(eff_c, EFF_DISP)) * wire + extra

    for k, sef, links, extra_head, tail in (
        ("2D-rep", 1.0, max(lk("d2c2", "Y"), lk("d2c2", "X")), 0, sm["misc"]),
        ("2D-tp", 1.0 / TP, max(lk("d2c2", "Y"), lk("d2c2", "X", ag + rs)), 0, sm["misc"]),
        ("own2D", 1.0 / TP, max(lk("d2c2", "Y"), lk("d2c2", "X", rs)), max(0, ag - head), sm["misc"]),
        ("1D", 1.0 / TP, max(lk("d1c1", "Y"), lk("d1c1", "X")), max(0, ag - head), sm["misc"] + rs_routed),
    ):
        dram = f + (se_w * sef + lat_w) / deff * 1e6
        out[k]["full"] = head + extra_head + max(dram, links) + tail
        out[k]["full_bound"] = "DRAM" if dram >= links else "link"
    return out


if __name__ == "__main__":
    chunks = [int(x) for x in (sys.argv[1:] or ["5120", "10240", "16384"])]
    cache = {}
    for name, p in M.items():
        for st in ("4x4", "8x4"):
            SP, TP, wtp, wsp = STAGES[st]
            if p["E"] % (SP * TP):
                continue
            for T in chunks:
                for lev in ((), ("tie",), ("tie", "fp8")):
                    key = (name, st, T, "tie" in lev)
                    if key not in cache:
                        cache[key] = sim(SP, TP, wtp, wsp, p["E"], p["K"], min(T, 8192), tie_split="tie" in lev)
                        cache[key] = {k: v * T / min(T, 8192) for k, v in cache[key].items()}
                    r = block(p, st, T, cache[key], lev)
                    tag = "+".join(lev) or "today"
                    print(
                        f"{name:14s} {st} T{T:6d} {tag:8s} | "
                        + " | ".join(
                            f"{k}: {v['serial']:5.0f} {v['ov']:5.0f} {v['full']:5.0f}{v['full_bound'][0]} (c {v['c']:4.0f} d {v['d']:4.0f})"
                            for k, v in r.items()
                        ),
                        flush=True,
                    )
