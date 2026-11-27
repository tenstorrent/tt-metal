# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: refit the quartic correction g(m) = 2^(m-1) / m on [1, 2) with G3, G2, G1 restricted to fp16
(SFPLOADI FLOATA: one load each instead of an fp32 USHORT + UPPER pair); G4 (held in L7 for a whole face)
and G0 (constant LReg 14) stay fp32. Minimax in relative error, linear programs over a dense grid.

    python3 fit_fp16_coeffs.py
"""
import itertools

import numpy as np
from scipy.optimize import linprog

M = np.linspace(1.0, 2.0, 40001)[:-1]
G = 2.0 ** (M - 1.0) / M
POW = np.stack([M**k for k in range(5)], axis=1)  # columns m^0 .. m^4


def minimax(fixed):
    """fixed: {k: value}. Minimize max |p/g - 1| over the free coefficients. Returns (coeffs, err)."""
    free = [k for k in range(5) if k not in fixed]
    base = sum(v * POW[:, k] for k, v in fixed.items()) if fixed else np.zeros_like(M)
    # variables: c_free..., t ; minimize t s.t. |(base + A c)/g - 1| <= t
    A = POW[:, free] / G[:, None]
    b = 1.0 - base / G
    n = len(free)
    A_ub = np.vstack([np.hstack([A, -np.ones((len(M), 1))]), np.hstack([-A, -np.ones((len(M), 1))])])
    b_ub = np.concatenate([b, -b])
    c = np.zeros(n + 1)
    c[-1] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * n + [(0, None)], method="highs")
    coeffs = dict(fixed)
    for i, k in enumerate(free):
        coeffs[k] = res.x[i]
    return coeffs, res.x[-1]


def err_of(coeffs):
    p = sum(np.float32(coeffs[k]).astype(np.float64) * POW[:, k] for k in range(5))
    return np.abs(p / G - 1).max()


def fp16_neighbors(v, span):
    h = np.float16(v)
    out = []
    for d in range(-span, span + 1):
        bits = np.array([h]).view(np.uint16)[0].astype(np.int32) + d
        out.append(float(np.array([bits], dtype=np.uint16).view(np.float16)[0]))
    return out


def main():
    free, e = minimax({})
    print(f"fp32 minimax quartic: rel err {e:.3e}  coeffs {[free[k] for k in range(5)]}")
    best = None
    for g3 in fp16_neighbors(free[3], 6):
        c3, _ = minimax({3: g3})
        for g2 in fp16_neighbors(c3[2], 6):
            c2, _ = minimax({3: g3, 2: g2})
            for g1 in fp16_neighbors(c2[1], 6):
                c1, e1 = minimax({3: g3, 2: g2, 1: g1})
                if best is None or e1 < best[1]:
                    best = (c1, e1)
    c, e = best
    print(f"fp16 G3,G2,G1 + fp32 G4,G0: rel err {e:.3e} (fp32-rounded eval {err_of(c):.3e})")
    for k in (4, 3, 2, 1, 0):
        v = np.float32(c[k])
        bits = np.array([v]).view(np.uint32)[0]
        f16 = np.array([np.float16(c[k])]).view(np.uint16)[0]
        exact16 = float(np.float16(c[k])) == float(v)
        print(f"  G{k} = {float(v)!r:24} fp32 0x{bits:08x}" + (f"  fp16 0x{f16:04x}" if exact16 else ""))


if __name__ == "__main__":
    main()
