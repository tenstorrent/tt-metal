# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What the tree's model picks per BH, the alternatives it nearly picked, and which of them to measure.

  python calib_picks.py --bh 4,8,12,16,24,32,48 [--grid 11x10] [--T 2048] [--within 0.04] [--collected x.json ...]
                        [--emit-rows prefix --method FORWARD_SUBSTITUTION --iters 5]

For each BH: the free pick (NV, NP or P, placement, depth, share, T model), then every candidate within --within of
it: the per-head geometries at each NV (the model's NP there and its neighbours, both depths) and the pools (NV 1/2,
the share around the model's, both depths). Every number comes from the binding (the C++ model), so this is exactly
what the op decides between. With --collected, the measured op of any matching row is shown next to the model's T.
With --emit-rows, the near-ties that have no measurement yet are printed as a rows file for calib_batch.sh: measure
those, then compare the pick against them (a pick within noise of its runner-up is fine; a runner-up that measures
clearly faster means a constant is off).
"""

import argparse
import json
import sys

from ttnn._ttnn.operations import transformer as _t

PER_HEAD, POOL, BOTH = 0, 1, 2
HK = {4: 4, 8: 4, 12: 4, 16: 4, 24: 8, 32: 8, 48: 16}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bh", default="4,8,12,16,24,32,48")
    ap.add_argument("--grid", default="11x10")
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--vt", type=int, default=4)
    ap.add_argument("--within", type=float, default=0.04, help="relative band around the pick's T")
    ap.add_argument("--collected", nargs="*", default=[])
    ap.add_argument("--emit-rows", default=None, help="prefix: print unmeasured near-ties as batch rows")
    ap.add_argument("--method", default="FORWARD_SUBSTITUTION")
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()
    gx, gy = (int(v) for v in a.grid.split("x"))
    NC = a.T // 32
    measured = {}
    for f in a.collected:
        for r in json.load(open(f)):
            if r.get("phased") or r["op"] != r["op"] or "num" not in r:
                continue
            key = (r["BH"], r["nv"], r["np"], r["PL"], r["nbuf"], r["num"] if r["PL"] == 2 else 0)
            measured.setdefault(key, []).append(r["op"])
    emit = []
    for bh in (int(v) for v in a.bh.split(",")):
        pick = _t.chunk_gdn_fused_geometry(gx, gy, bh, NC, a.vt)
        nv, np_, pl, d, t_pick, t_ph, pays, num = pick
        if nv == 0:
            print(f"BH={bh}: no fused geometry fits {a.grid}; phased {t_ph:.0f}")
            continue
        cands = {}

        def add(nv, np_, pl, d, num, t):
            cands[(nv, np_, pl, d, num)] = t

        for cnv in (1, 2, 4, 8):
            if a.vt % cnv:
                continue
            best = _t.chunk_gdn_fused_geometry(gx, gy, bh, NC, a.vt, cnv, 0, 0, PER_HEAD)
            if best[0] == cnv and best[2] != 2:
                for cnp in (best[1] - 1, best[1], best[1] + 1):
                    for cd in (2, 3):
                        if cnp < 1:
                            continue
                        g = _t.chunk_gdn_fused_geometry(gx, gy, bh, NC, a.vt, cnv, cnp, cd, PER_HEAD)
                        if g[0] == cnv and g[1] == cnp:
                            add(cnv, cnp, g[2], cd, 0, g[4])
            if cnv <= 2 and bh * cnv < gx * gy:
                P = min(gx * gy - bh * cnv, bh * NC)
                if _t.chunk_gdn_fused_pool_feasible(gx, gy, bh, cnv, P):
                    nph = _t.chunk_gdn_fused_pool_home_producers(gx, gy, bh, cnv, P)
                    nx = P - bh * nph
                    if nx:
                        pb = _t.chunk_gdn_fused_geometry(gx, gy, bh, NC, a.vt, cnv, P, 0, POOL)
                        for cnum in sorted({pb[7] + k for k in range(-3, 4)} | {nx}):
                            if 0 <= cnum <= nx:
                                for cd in (2, 3):
                                    g = _t.chunk_gdn_fused_geometry(gx, gy, bh, NC, a.vt, cnv, P, cd, POOL, cnum / P)
                                    add(cnv, P, 2, cd, g[7], g[4])
        hk = HK.get(bh, max(4, bh // 4))
        desc = lambda nv, np_, pl, d, num: (
            f"pool NV{nv} P{np_} share {num}/{np_} d{d}"
            if pl == 2
            else f"{'row-local' if pl == 1 else 'row-major'} NV{nv} NP{np_} d{d}"
        )
        print(f"\nBH={bh}: pick {desc(nv, np_, pl, d, num)}  T {t_pick:.1f} us (phased {t_ph:.0f}, fused pays: {pays})")
        print(f"   {'candidate':40s} {'model':>7s} {'vs pick':>8s} {'measured':>18s}")
        for key, t in sorted(cands.items(), key=lambda kv: kv[1]):
            if t > t_pick * (1 + a.within):
                continue
            m = measured.get((bh,) + key)
            ms = ", ".join(f"{v:.1f}" for v in m) if m else "-"
            mark = "  <= pick" if key == (nv, np_, pl, d, num) else ""
            print(f"   {desc(*key):40s} {t:7.1f} {(t / t_pick - 1) * 100:+7.1f}% {ms:>18s}{mark}")
            if a.emit_rows and not m:
                cnv, cnp, cpl, cd, cnum = key
                args = f"--method {a.method} --T {a.T} --iters {a.iters} --hk {hk} --hv {bh}"
                if cpl == 2:
                    emit.append(
                        f"{a.emit_rows}_bh{bh}_pool_nv{cnv}p{cnp}_s{cnum}_d{cd} calib_capture.py {args} --pool --nv {cnv} --np {cnp} --share {cnum / cnp:.4f} --nbuf {cd}"
                    )
                else:
                    emit.append(
                        f"{a.emit_rows}_bh{bh}_nv{cnv}np{cnp}_d{cd} calib_capture.py {args} --nv {cnv} --np {cnp} --rl {cpl} --nbuf {cd}"
                    )
    if a.emit_rows:
        print(f"\n# rows to measure (near-ties without a measurement), {len(emit)}:")
        for e in emit:
            print(e)


if __name__ == "__main__":
    main()
