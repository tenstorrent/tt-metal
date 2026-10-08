# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The fused cost model (a Python port of choose_fused_geometry's t_fused_us) against collected rows.

  python calib_model.py collected.json [more.json ...] [k=v ...] [--check] [--terms] [--fit k1,k2,..] [--only a,b] [--bh 12,16]
                        [--emit] [--table] [--grid 11x10]

  (no flag)   the error table: measured op vs the model at the constants in K (the shipped values, overridable as k=v)
  --check     the port against the tree's C++ through the binding, row by row (pinned geometry): must print max diff 0.0
  --terms     the directly measured terms per row (item, period, fill, step, pace, tail, skew, waits, regime) and their
              direct fits: W from the production-bound rows' VALID period, step per Vtl from the chain-bound rows' pace,
              the chain slope, fill = A + min(BH,24)*(B + C*P) by least squares, skew/tail vs BH
  --fit       coordinate descent over the named constants minimising the mean |relative error| of the selected rows
  --emit      print the C++ constexpr block and the Python-oracle block for the constants in K (after a fit)
Rows: the JSON of calib_collect.py (label, op, BH, NC, nv, np, PL, nbuf, share, nph, nx, num, den, phased, ...).
"""

import json
import sys

# The shipped constants (chunk_gdn_device_operation.cpp, anonymous namespace; mirrored in test_chunk_gdn_fused_geometry.py).
K = dict(
    W=15.98,  # kWpUs: producer item period, flat over BH and the producer count
    step1=3.17,
    step2=2.69,
    step4=4.45,  # t_step_us(Vtl): receiver step per V-slice width
    pool_step1=4.25,  # t_step_us(1, pooled): a pool's extras lengthen the Vtl=1 round trip
    slope=0.012,  # kChainSlopeUs: chunk period growth per head
    skewA=4.0,
    skewB=0.5,  # kSkewAUs/BUs: slowest chain behind the median
    tail=4.0,  # kTailUs
    fillA=19.5,
    fillB=0.72,
    fillC=1.0 / 144.0,  # fill = A + min(BH,24) * (B + C * producers)
    rowmajor=8.4,  # kRowMajorPaceUs
    peak=0.08,
    widthD2=0.25,
    widthD3=0.15,  # kBalancePeak / kBalanceWidthD2 / D3 (chain vs producers, Vtl <= 2)
    pool_peak=0.25,
    pool_width=0.25,  # kPoolBalancePeak / Width (home vs extra producers)
    pool_start=20.0,
    pool_start_share=0.25,  # kPoolStartUs * min(1, share / kPoolStartShare)
    pool_skew=4.0,  # kPoolSkewUs
    pool_credit_d2=0.025,  # kPoolCreditD2Us per head per step (pool, depth 2, Vtl <= 2)
)
PHASED = [(4, 359.8), (8, 461.5), (12, 603.0), (16, 769.4), (32, 1278.8), (48, 1907.9)]
CPP_NAMES = {
    "W": "kWpUs",
    "slope": "kChainSlopeUs",
    "skewA": "kSkewAUs",
    "skewB": "kSkewBUs",
    "tail": "kTailUs",
    "fillA": "kFillAUs",
    "fillB": "kFillBUs",
    "fillC": "kFillCUs",
    "rowmajor": "kRowMajorPaceUs",
    "peak": "kBalancePeak",
    "widthD2": "kBalanceWidthD2",
    "widthD3": "kBalanceWidthD3",
    "pool_peak": "kPoolBalancePeak",
    "pool_width": "kPoolBalanceWidth",
    "pool_start": "kPoolStartUs",
    "pool_start_share": "kPoolStartShare",
    "pool_skew": "kPoolSkewUs",
    "pool_credit_d2": "kPoolCreditD2Us",
}
PY_NAMES = {
    "W": "_W_P_US",
    "slope": "_CHAIN_SLOPE_US",
    "tail": "_TAIL_US",
    "rowmajor": "_ROW_MAJOR_PACE_US",
    "pool_skew": "_POOL_SKEW_US",
    "pool_credit_d2": "_POOL_CREDIT_D2_US",
    "pool_step1": "_T_STEP_POOL_VTL1_US",
}


def extras_before(bh, num, den, h, c):
    return (c * num * bh + (bh - 1 - h) * den) // (den * bh)


def pool_load(bh, nc, nph, nx, num, den):
    """Items of the busiest home producer and of the busiest extra (the shared item map)."""
    if nx == 0:
        return -(-nc // nph), 0
    nh = max(nc - extras_before(bh, num, den, h, nc) for h in range(bh))
    ne = (nc * num * bh) // den
    return -(-nh // nph), -(-ne // nx)


def balance(a, b, width, peak):
    if a <= 0 or b <= 0:
        return 0.0
    m = max(a, b)
    return peak * max(0.0, 1.0 - abs(a - b) / (width * m))


def t_fused(r, K=K, vt=4):
    """t_fused_us of chunk_gdn_device_operation.cpp for one collected row; returns (T, terms)."""
    bh, nc, nv, pl, depth = r["BH"], r.get("NC", 64), r["nv"], r["PL"], r["nbuf"]
    vtl = vt // nv
    nph, nx, num, den = r["nph"], r["nx"], r["num"], r["den"]
    producers = r["np"] if pl == 2 else bh * r["np"]
    pooled = nx > 0
    share = num / den if pooled else 0.0
    n_home, n_extra = pool_load(bh, nc, nph, nx, num, den)
    step = K["pool_step1"] if (pooled and vtl == 1) else K[f"step{vtl}"]
    pace = step + K["slope"] * bh
    if pooled and vtl <= 2 and depth <= 2:
        pace += K["pool_credit_d2"] * bh
    H = (n_home - 1) * K["W"]
    X = (n_extra - 1) * K["W"] if n_extra else 0.0
    if pl == 0 and max(H, X) < 2.0 * (nc - 1) * pace:
        pace = max(pace, K["rowmajor"])
    C = (nc - 1) * pace + K["skewA"] + K["skewB"] * bh + (K["pool_skew"] if pooled else 0.0)
    pen = start = 0.0
    if vtl <= 2:
        pen = balance(C, H, K["widthD2"] if depth <= 2 else K["widthD3"], K["peak"])
        if pooled:
            m = max(C, H, X)
            g = max(0.0, 1.0 - (m - max(H, X)) / (K["pool_width"] * m))
            pen = max(pen, g * balance(H, X, K["pool_width"], K["pool_peak"]))
            start = K["pool_start"] * min(1.0, share / K["pool_start_share"])
    fill = K["fillA"] + min(bh, 24) * (K["fillB"] + K["fillC"] * producers)
    T = fill + max(C, H + start, X) * (1.0 + pen) + K["tail"]
    bound = "C" if C >= max(H + start, X) else ("H" if H + start >= X else "X")
    return T, dict(
        n_home=n_home,
        n_extra=n_extra,
        fill=round(fill, 1),
        C=round(C, 1),
        H=round(H, 1),
        X=round(X, 1),
        start=round(start, 1),
        pen=round(pen, 3),
        bound=bound,
    )


def t_phased(bh, nc, table=PHASED):
    x = min(bh, table[-1][0])
    i = 0
    while i + 2 < len(table) and x > table[i + 1][0]:
        i += 1
    (x0, y0), (x1, y1) = table[i], table[i + 1]
    t = y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    if bh > table[-1][0]:
        t *= bh / table[-1][0]
    return t * nc / 64.0


def main():
    argv = sys.argv[1:]
    rows, fit, only, bhs = [], [], [], []
    gx, gy = 11, 10
    opts = {"--check": False, "--terms": False, "--emit": False, "--table": False}
    i = 0
    while i < len(argv) and argv[i].endswith(".json"):
        rows += json.load(open(argv[i]))
        i += 1
    while i < len(argv):
        a = argv[i]
        if a == "--fit":
            fit = argv[i + 1].split(",")
            i += 2
        elif a == "--only":
            only = argv[i + 1].split(",")
            i += 2
        elif a == "--bh":
            bhs = [int(v) for v in argv[i + 1].split(",")]
            i += 2
        elif a == "--grid":
            gx, gy = (int(v) for v in argv[i + 1].split("x"))
            i += 2
        elif a in opts:
            opts[a] = True
            i += 1
        else:
            k, v = a.split("=")
            if k not in K:
                sys.exit(f"unknown constant {k}; known: {', '.join(K)}")
            K[k] = float(v)
            i += 1
    usable = lambda r: (not r["phased"]) and r["op"] == r["op"] and "nph" in r and r.get("nbuf", 0) > 0
    fused = [
        r
        for r in rows
        if usable(r) and (not only or any(s in r["label"] for s in only)) and (not bhs or r["BH"] in bhs)
    ]
    phased = [r for r in rows if r["phased"] and r["op"] == r["op"]]
    if not fused and not opts["--emit"]:
        sys.exit("no usable fused rows (need op, nph/nx/num/den from calib_collect.py and a depth)")

    if opts["--check"]:
        from ttnn._ttnn.operations import transformer as _t

        worst = 0.0
        for r in fused:
            cand = 1 if r["PL"] == 2 else 0
            share = (r["num"] / r["den"]) if r["PL"] == 2 else -1.0
            got = _t.chunk_gdn_fused_geometry(
                gx, gy, r["BH"], r.get("NC", 64), 4, r["nv"], r["np"], r["nbuf"], cand, share
            )
            t, _ = t_fused(r)
            d = abs(got[4] - t)
            worst = max(worst, d)
            flag = "" if d < 0.05 else "   <-- DIFFERS"
            print(f"{r['label']:28s} cpp {got[4]:8.2f} port {t:8.2f} diff {d:6.2f}{flag}")
        print(
            f"max |cpp - port| over {len(fused)} rows: {worst:.3f} us  ({'port matches the tree' if worst < 0.05 else 'PORT OUT OF DATE: update K / t_fused'})"
        )
        return

    if opts["--terms"]:
        import numpy as np

        print(
            f"{'label':28s} {'BH':>3s} {'nv':>2s} {'np':>3s} {'PL':>2s} {'d':>1s} {'share':>6s} {'op':>7s} | {'nh':>2s} {'nx':>2s} {'item':>5s} {'per':>5s} {'fill':>5s} {'step':>5s} {'pace':>5s} {'last':>6s} {'skew':>5s} {'swait':>5s} {'txcr':>5s} {'txcb':>5s} regime"
        )
        for r in fused:
            s = "-" if r["share"] is None else f"{r['share']:.3f}"
            print(
                f"{r['label']:28s} {r['BH']:3d} {r['nv']:2d} {r['np']:3d} {r['PL']:2d} {r['nbuf']:1d} {s:>6s} {r['op']:7.1f} | {r['n_home']:2d} {r['n_extra']:2d} "
                f"{r['item']:5.2f} {r['per']:5.2f} {r['fill']:5.1f} {r['step']:5.2f} {r['pace']:5.2f} {r['last']:6.1f} {r['skew']:5.1f} {r['swait']:5.2f} {r['txcr']:5.2f} {r['txcb']:5.2f} {r['regime']}"
            )
        print("\ndirect estimates (hints; the residual fit decides):")
        prod = [r for r in fused if r["regime"] == "production-bound"]
        chain = [r for r in fused if r["regime"] == "chain-bound" and r["PL"] != 2]  # per-head rows: no pool terms
        if prod:
            print(
                f"  W (kWpUs) = VALID period on production-bound rows: median {np.median([r['per'] for r in prod]):.2f} "
                f"(min {min(r['per'] for r in prod):.2f} max {max(r['per'] for r in prod):.2f}, n={len(prod)}); "
                f"prep_item zone median over all rows {np.median([r['item'] for r in fused]):.2f}"
            )
        for vtl in (1, 2, 4):
            rs = sorted((r for r in chain if 4 // r["nv"] == vtl and r["nbuf"] >= 3), key=lambda r: r["BH"])
            if not rs:
                continue
            pairs = ", ".join(f"BH{r['BH']}:{r['pace']:.2f}" for r in rs)
            line = f"  step{vtl}: per-head chain-bound depth>=3 pace by BH: {pairs}; scan_step zone median {np.median([r['step'] for r in rs]):.3f}"
            if len({r["BH"] for r in rs}) >= 3:
                A = np.array([[1.0, r["BH"]] for r in rs])
                y = np.array([r["pace"] for r in rs])
                c, *_ = np.linalg.lstsq(A, y, rcond=None)
                line += f"; fit pace = {c[0]:.3f} + {c[1]:.4f}*BH (step{vtl} + kChainSlopeUs)"
            print(line)
        F = [(r["BH"], r["np"] if r["PL"] == 2 else r["BH"] * r["np"], r["fill"]) for r in fused]
        A = np.array([[1.0, min(b, 24), min(b, 24) * p] for b, p, f in F])
        y = np.array([f for b, p, f in F])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ c
        print(
            f"  fill = {c[0]:.1f} + min(BH,24) * ({c[1]:.3f} + {c[2] * 144:.2f}/144 * P)   rms {np.sqrt((res ** 2).mean()):.1f}  max |res| {abs(res).max():.1f}  (n={len(F)})"
        )
        if chain:
            by_bh = {}
            for r in chain:
                by_bh.setdefault(r["BH"], []).append(r["op"] - r["last"])
            print(
                "  op - median chain end (= kSkewAUs + kTailUs + kSkewBUs*BH) on per-head chain-bound rows, median per BH: "
                + ", ".join(f"BH{b}:{np.median(v):.1f}" for b, v in sorted(by_bh.items()))
            )
        rm = [r for r in fused if r["PL"] == 0]
        if rm:
            print("row-major pace: " + ", ".join(f"{r['pace']:.2f}" for r in rm))
        if phased:
            print("phased: " + ", ".join(f"BH{r['BH']} {r['op']:.1f}" for r in phased))
        return

    def score(K):
        return sum(abs(t_fused(r, K)[0] - r["op"]) / r["op"] for r in fused) / len(fused)

    if fit:
        best = score(K)
        print(f"start mean|err| {best * 100:.2f}% over {len(fused)} rows")
        steps = {k: max(abs(K[k]) * 0.1, 0.005) for k in fit}
        for it in range(80):
            moved = False
            for k in fit:
                for sgn in (+1, -1):
                    K2 = dict(K)
                    K2[k] = K[k] + sgn * steps[k]
                    if K2[k] < 0:
                        continue
                    s = score(K2)
                    if s < best - 1e-7:
                        K.update(K2)
                        best, moved = s, True
            if not moved:
                for k in fit:
                    steps[k] /= 2
                if max(steps.values()) < 1e-4:
                    break
        print("fitted:", {k: round(K[k], 4) for k in fit}, f"mean|err| {best * 100:.2f}%")

    if opts["--emit"]:
        print("// C++ (chunk_gdn_device_operation.cpp)")
        for k, n in CPP_NAMES.items():
            print(f"constexpr float {n} = {K[k]:.4g}f;")
        print(
            f"// t_step_us: case 1: return pooled ? {K['pool_step1']:.2f}f : {K['step1']:.2f}f; case 2: return {K['step2']:.2f}f; case 4: return {K['step4']:.2f}f;"
        )
        print("# Python oracle (test_chunk_gdn_fused_geometry.py)")
        print(
            f"_W_P_US = {K['W']}\n_T_STEP_US = {{1: {K['step1']}, 2: {K['step2']}, 4: {K['step4']}}}\n_T_STEP_POOL_VTL1_US = {K['pool_step1']}"
        )
        print(
            f"_CHAIN_SLOPE_US = {K['slope']}\n_SKEW_A_US, _SKEW_B_US = {K['skewA']}, {K['skewB']}\n_TAIL_US = {K['tail']}"
        )
        print(
            f"_FILL_A_US, _FILL_B_US, _FILL_C_US = {K['fillA']}, {K['fillB']}, {K['fillC']}\n_ROW_MAJOR_PACE_US = {K['rowmajor']}"
        )
        print(f"_BALANCE_PEAK, _BALANCE_WIDTH_D2, _BALANCE_WIDTH_D3 = {K['peak']}, {K['widthD2']}, {K['widthD3']}")
        print(
            f"_POOL_BALANCE_PEAK, _POOL_BALANCE_WIDTH = {K['pool_peak']}, {K['pool_width']}\n_POOL_START_US, _POOL_START_SHARE = {K['pool_start']}, {K['pool_start_share']}"
        )
        print(f"_POOL_SKEW_US = {K['pool_skew']}\n_POOL_CREDIT_D2_US = {K['pool_credit_d2']}")
        if not fused:
            return

    errs = []
    print(
        f"{'label':28s} {'BH':>3s} {'nv':>2s} {'np':>3s} {'PL':>2s} {'d':>1s} {'share':>6s} {'meas':>7s} {'model':>7s} {'err':>6s}  terms"
    )
    for r in fused:
        t, info = t_fused(r)
        e = (t - r["op"]) / r["op"] * 100
        errs.append(abs(e))
        s = "-" if r["share"] is None else f"{r['share']:.3f}"
        print(
            f"{r['label']:28s} {r['BH']:3d} {r['nv']:2d} {r['np']:3d} {r['PL']:2d} {r['nbuf']:1d} {s:>6s} {r['op']:7.1f} {t:7.1f} {e:+5.1f}%  {info}"
        )
    print(
        f"mean |err| {sum(errs) / len(errs):.2f}%  max {max(errs):.1f}%  rows > 5%: {sum(e > 5 for e in errs)} of {len(errs)}"
    )
    for r in phased:
        print(f"phased BH={r['BH']:3d} measured {r['op']:7.1f}  table {t_phased(r['BH'], r.get('NC', 64)):7.1f}")


if __name__ == "__main__":
    main()
