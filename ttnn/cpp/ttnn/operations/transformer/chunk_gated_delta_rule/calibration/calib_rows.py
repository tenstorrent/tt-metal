# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The measurement matrix of one calibration, as a rows file for calib_batch.sh.

  python calib_rows.py --prefix c1 --bh 4,8,12,16,24,32,48 [--sets phased,auto,perhead,pool,share,rowmajor]
                       [--grid 11x10] [--T 2048] [--method FORWARD_SUBSTITUTION] [--iters 5] [--repeats 3] > c1_rows.txt

One line per capture: "label script args". The geometries come from the tree's own model (the ttnn binding of the
worktree in TT_METAL_HOME), so the matrix follows the current picks:
  phased    the two-phase reference (prep + scan) per BH                 -> the phased table
  auto      the op's own pick (no program config), --repeats captures    -> the headline, the pick to confirm
  perhead   per NV (1, 2, 4): the largest NP that fits and the model's NP at that NV with its neighbours, depths 2 and 3
            -> item, step per Vtl, slope, fill, skew, the chain/production balance bump
  pool      per NV (1, 2): the full pool at the balanced share, depths 2 and 3  -> pool step, pool skew, credit term
  share     at the model's pool pick: the share from the balance down (balanced, -1, -2, -3, -5, -8, half, quarter, 0)
            -> the home/extra bump, the pool start term, the share choice
  rowmajor  the per-head pick with row_local=0 (one row per BH)        -> kRowMajorPaceUs
Captures take ~10 s each once the JIT cache is warm; a full matrix for seven BH values is ~190 rows.
"""

import argparse
import sys

from ttnn._ttnn.operations import transformer as _t

PER_HEAD, POOL, BOTH = 0, 1, 2
HK = {4: 4, 8: 4, 12: 4, 16: 4, 24: 8, 32: 8, 48: 16}  # key heads per BH used by the existing captures (GQA shape only)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--bh", default="4,8,12,16,24,32,48")
    ap.add_argument("--sets", default="phased,auto,perhead,pool,share")
    ap.add_argument("--grid", default="11x10")
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--vt", type=int, default=4)
    ap.add_argument("--method", default="FORWARD_SUBSTITUTION")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3, help="captures of the op's own pick per BH")
    a = ap.parse_args()
    gx, gy = (int(v) for v in a.grid.split("x"))
    NC = a.T // 32
    sets = set(a.sets.split(","))
    bhs = [int(v) for v in a.bh.split(",")]
    base = f"--method {a.method} --T {a.T} --iters {a.iters}"
    rows = []

    def row(label, args, script="calib_capture.py"):
        rows.append(f"{a.prefix}_{label} {script} {args}")

    for bh in bhs:
        hk = HK.get(bh, max(4, bh // 4))
        if "phased" in sets:
            row(f"ph_bh{bh}", f"--phased --hk {max(1, bh // 4)} --hv {bh} --T {a.T} --iters {a.iters}")
        if "auto" in sets:
            for i in range(1, a.repeats + 1):
                row(f"auto_bh{bh}_A{i}", f"{base} --hk {hk} --hv {bh}")
        pick = _t.chunk_gdn_fused_geometry(gx, gy, bh, NC, a.vt)
        if "perhead" in sets:
            seen = set()
            for nv in (1, 2, 4):
                if a.vt % nv:
                    continue
                np_max = 0
                for np_ in range(1, gx):
                    if bh * (nv + np_) <= gx * gy and _t.chunk_gdn_fused_row_local_feasible(
                        gx, gy, bh, nv, min(np_, NC)
                    ):
                        np_max = min(np_, NC)
                if np_max == 0:
                    continue
                best = _t.chunk_gdn_fused_geometry(gx, gy, bh, NC, a.vt, nv, 0, 0, PER_HEAD)
                cands = {np_max}
                if best[0] == nv and best[2] == 1:
                    cands |= {best[1], best[1] - 1, best[1] + 1}
                for np_ in sorted(c for c in cands if 1 <= c <= np_max):
                    for d in (2, 3):
                        if (nv, np_, d) in seen:
                            continue
                        seen.add((nv, np_, d))
                        row(
                            f"bh{bh}_nv{nv}np{np_}_d{d}",
                            f"{base} --hk {hk} --hv {bh} --nv {nv} --np {np_} --rl 1 --nbuf {d}",
                        )
        if "rowmajor" in sets:
            ph = _t.chunk_gdn_fused_geometry(gx, gy, bh, NC, a.vt, 0, 0, 0, PER_HEAD)
            if ph[0] and bh <= (gx // ph[0]) * gy:
                row(
                    f"bh{bh}_nv{ph[0]}np{ph[1]}_rm",
                    f"{base} --hk {hk} --hv {bh} --nv {ph[0]} --np {ph[1]} --rl 0 --nbuf {ph[3]}",
                )
        if "pool" in sets or "share" in sets:
            for nv in (1, 2):
                if a.vt % nv or bh * nv >= gx * gy:
                    continue
                P = min(gx * gy - bh * nv, bh * NC)
                if not _t.chunk_gdn_fused_pool_feasible(gx, gy, bh, nv, P):
                    continue
                nph = _t.chunk_gdn_fused_pool_home_producers(gx, gy, bh, nv, P)
                nx = P - bh * nph
                if nx == 0:
                    continue
                if "pool" in sets:
                    for d in (2, 3):
                        row(
                            f"bh{bh}_pool_nv{nv}p{P}_bal_d{d}",
                            f"{base} --hk {hk} --hv {bh} --pool --nv {nv} --np {P} --share {nx / P:.4f} --nbuf {d}",
                        )
                if "share" in sets and pick[2] == 2 and pick[0] == nv and pick[1] == P:
                    nums = sorted(
                        {nx, nx - 1, nx - 2, nx - 3, nx - 5, nx - 8, nx // 2, nx // 4, 0, pick[7]}
                        - {n for n in range(-100, 0)}
                    )
                    for num in nums:
                        for d in (2, 3) if num in (nx, pick[7]) else (pick[3],):
                            row(
                                f"bh{bh}_pool_nv{nv}p{P}_s{num}_d{d}",
                                f"{base} --hk {hk} --hv {bh} --pool --nv {nv} --np {P} --share {num / P:.4f} --nbuf {d}",
                            )
    seen = set()
    print(f"# calib rows: prefix {a.prefix}, grid {a.grid}, T={a.T} (NC={NC}), sets {','.join(sorted(sets))}")
    print("# label script args   (share values are num/P; the balanced share is NX/P)")
    for r in rows:
        if r.split()[0] in seen:
            continue
        seen.add(r.split()[0])
        print(r)
    print(f"# {len(seen)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
