"""Minimax (relative error) fit of 2^r on r in [-0.5, 0.5] with c0 = 1, via LP. Prints coefficients scaled to r' = r / SCALE."""
import sys
import numpy as np
from scipy.optimize import linprog

import os

SCALE = float(os.environ.get("FIT_SCALE", "64"))


def fit(deg, n=4001, zero_mean=False):
    r = np.linspace(-0.5, 0.5, n)
    f = np.exp2(r)
    # p(r) = 1 + sum_{k=1..deg} c_k r^k ; minimize t s.t. |p/f - 1| <= t
    A = np.stack([r**k / f for k in range(1, deg + 1)], axis=1)
    b0 = 1.0 / f - 1.0  # p/f - 1 = A c + b0
    nv = deg + 1
    cost = np.zeros(nv)
    cost[-1] = 1
    A_ub = np.vstack([np.hstack([A, -np.ones((n, 1))]), np.hstack([-A, -np.ones((n, 1))])])
    b_ub = np.concatenate([-b0, b0])
    eq = {}
    if zero_mean:
        eq = dict(A_eq=np.hstack([A.mean(axis=0), [0.0]])[None, :], b_eq=[-b0.mean()])
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * nv, method="highs", **eq)
    c = res.x[:deg]
    c32 = np.float32(c)
    p = 1 + sum(np.float64(c32[k - 1]) * r**k for k in range(1, deg + 1))
    rel = p / f - 1
    return c, rel


for deg, zm in ((3, False), (3, True), (4, False), (5, False)):
    c, rel = fit(deg, zero_mean=zm)
    print(f"deg {deg} zero_mean={zm}: max rel {np.abs(rel).max():.3e} mean rel {rel.mean():.3e}")
    for k, ck in enumerate(c, 1):
        print(f"  c{k} = {ck:.10e}   scaled(r'=r/{SCALE:g}) = {np.float32(ck * SCALE**k):.10e}")


# --- bf16-constrained refit (c2.. rounded to bf16 in the r' = r/SCALE domain so each loads with one SFPLOADI) ---


def bf16_neighbors(v, k=3):
    import torch

    b = torch.tensor([v], dtype=torch.float32).to(torch.bfloat16).view(torch.int16).item()
    return [torch.tensor([b + d], dtype=torch.int16).view(torch.bfloat16).float().item() for d in range(-k, k + 1)]


def fit_fixed(deg, fixed, zero_mean, n=4001):
    """fixed: {k: value in r' domain}; free coefficients fitted (r domain). Returns (max_rel, mean_rel, coeffs_rprime)."""
    r = np.linspace(-0.5, 0.5, n)
    f = np.exp2(r)
    free = [k for k in range(1, deg + 1) if k not in fixed]
    base = 1.0 + sum(np.float64(np.float32(v)) / SCALE**k * r**k for k, v in fixed.items())
    A = np.stack([r**k / f for k in free], axis=1)
    b0 = base / f - 1.0
    nv = len(free) + 1
    cost = np.zeros(nv)
    cost[-1] = 1
    A_ub = np.vstack([np.hstack([A, -np.ones((n, 1))]), np.hstack([-A, -np.ones((n, 1))])])
    b_ub = np.concatenate([-b0, b0])
    eq = dict(A_eq=np.hstack([A.mean(axis=0), [0.0]])[None, :], b_eq=[-b0.mean()]) if zero_mean else {}
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * nv, method="highs", **eq)
    coeffs = dict(fixed)
    for k, c in zip(free, res.x[: len(free)]):
        coeffs[k] = float(np.float32(c * SCALE**k))
    p = 1 + sum(np.float64(np.float32(coeffs[k])) / SCALE**k * r**k for k in range(1, deg + 1))
    rel = p / f - 1
    return np.abs(rel).max(), rel.mean(), coeffs


def fit_bf16(deg, zero_mean):
    fixed = {}
    for k in range(deg, 1, -1):  # round top-down, refit the rest each time
        _, _, c = fit_fixed(deg, fixed, zero_mean)
        best = min(bf16_neighbors(c[k]), key=lambda v: fit_fixed(deg, {**fixed, k: v}, zero_mean)[0])
        fixed[k] = best
    return fit_fixed(deg, fixed, zero_mean)


print("\n# bf16 c2.. (r' domain), fp32 c1")
for deg in (3, 4):
    for zm in (False, True):
        mx, mn, c = fit_bf16(deg, zm)
        print(
            f"deg {deg} zero_mean={zm}: max rel {mx:.3e} mean rel {mn:.3e}  "
            + "  ".join(f"c{k}={c[k]:.10e}" for k in sorted(c))
        )
