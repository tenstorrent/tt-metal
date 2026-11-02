# Offline coefficient fitting (numpy only, no device) for the gelu_alt_approx candidates.
#   python fit.py
import numpy as np

K = np.sqrt(2 / np.pi)
LOG2E = 1.4426950408889634


def gelu(x):
    return 0.5 * x * (1 + np.tanh(K * (x + 0.044715 * x**3)))


def log2_s(a):
    # S(a) = sigmoid(-2u(a)); log2 S = -log2(1 + 2^v), v = 2u log2e
    v = 2 * K * (a + 0.044715 * a**3) * LOG2E
    return -(np.logaddexp(0, v * np.log(2)) / np.log(2))


def lawson(fx, x, deg, w=None, iters=200, basis=None):
    """Weighted minimax polynomial fit via Lawson IRLS. Returns coeffs low->high."""
    if w is None:
        w = np.ones_like(x)
    V = np.vstack([x**i for i in range(deg + 1)]).T if basis is None else basis(x)
    lw = np.ones_like(x) / len(x)
    for _ in range(iters):
        sw = np.sqrt(lw) * w
        c, *_ = np.linalg.lstsq(V * sw[:, None], fx * sw, rcond=None)
        err = np.abs((V @ c - fx) * w)
        lw = lw * err
        lw /= lw.sum()
    return c


def bf16_exact(c):
    return (
        float(np.float32(c).view(np.uint32) & 0xFFFF0000)
        if False
        else float((np.array([c], dtype=np.float32).view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)[0])
    )


def fit_log2s(deg, afit, wtail=0.05):
    a = np.linspace(0, afit, 20000)
    f = log2_s(a)
    # precision weight: tight where |gelu(-a)| >= ~1e-4, loose deeper in the tail
    w = np.where(a <= 4.0, 1.0, wtail)
    c = lawson(f, a, deg, w)
    return c


def exp2_fit(deg, lo, hi, zero_mean=True):
    f = np.linspace(lo, hi, 20000)
    # relative error fit: weight by 1/2^f
    c = lawson(2.0**f, f, deg, w=2.0**-f)
    return c


if __name__ == "__main__":
    for afit in [4.5, 5.0, 5.5, 6.0]:
        for deg in [4, 5, 6, 7]:
            c = fit_log2s(deg, afit)
            a = np.linspace(0, afit, 100000)
            e = np.polyval(c[::-1], a) - log2_s(a)
            m = a <= 3.5
            print(
                f"afit {afit} deg {deg}: max|dP| a<=3.5 {np.abs(e[m]).max():.2e}  a<=4 {np.abs(e[a<=4]).max():.2e}"
                f" all {np.abs(e).max():.2e}  lead {c[-1]:+.3e}"
            )
    for lo, hi in [(0, 1), (-0.5, 1), (-0.5, 0.5)]:
        for deg in [2, 3, 4]:
            c = exp2_fit(deg, lo, hi)
            f = np.linspace(lo, hi, 100000)
            r = np.polyval(c[::-1], f) / 2.0**f - 1
            print(f"exp2 [{lo},{hi}] deg {deg}: max rel {np.abs(r).max():.2e}")


def bf16_round(c):
    """Round an fp32 to the nearest BF16 (RNE) and return it as a Python float."""
    b = np.array([c], dtype=np.float32).view(np.uint32)[0]
    b = (int(b) + 0x7FFF + ((int(b) >> 16) & 1)) & 0xFFFF0000
    return float(np.array([b], dtype=np.uint32).view(np.float32)[0])


def fit_exp2_c0_one(lo, hi, deg, n_bf16):
    """2^f ~ 1 + c1 f + ... + c_deg f^deg on [lo, hi], relative error, minimax then zero-mean on c1.
    The top n_bf16 coefficients are rounded to BF16 (one SFPLOADI each) with refits of the rest."""
    f = np.linspace(lo, hi, 40001)
    target = 2.0**f
    w = 1 / target
    fixed = {}
    for _ in range(n_bf16 + 1):
        free = [k for k in range(1, deg + 1) if k not in fixed]
        resid = target - 1 - sum(fixed[k] * f**k for k in fixed)
        V = np.vstack([f**k for k in free]).T
        lw = np.ones_like(f) / len(f)
        for _ in range(300):
            sw = np.sqrt(lw) * w
            c, *_ = np.linalg.lstsq(V * sw[:, None], resid * sw, rcond=None)
            err = np.abs((V @ c - resid) * w)
            lw = lw * (err + 1e-30)
            lw /= lw.sum()
        coeffs = dict(zip(free, c))
        if len(fixed) == n_bf16:
            break
        top = max(free)
        fixed[top] = bf16_round(coeffs[top])
    coeffs.update(fixed)

    # zero-mean: shift c1 so the mean relative error over uniform f is 0
    def rel(cs):
        p = 1 + sum(cs[k] * f**k for k in cs)
        return p / target - 1

    r = rel(coeffs)
    m1 = np.mean(f / target)
    coeffs[1] -= np.mean(r) / m1
    r = rel(coeffs)
    c = [1.0] + [float(np.float32(coeffs[k])) for k in range(1, deg + 1)]
    return c, np.abs(r).max(), np.mean(r)


def report_exp2():
    for lo, hi, name in [(0.0, 1.0, "T21 [0,1)"), (-0.5, 0.5, "MAG RNE [-0.5,0.5]"), (-0.5, 1.0, "MAG any [-0.5,1)")]:
        for deg in [3, 4]:
            c, mx, mn = fit_exp2_c0_one(lo, hi, deg, deg - 1)
            p1 = sum(ck * hi**k for k, ck in enumerate(c))
            print(f"{name} deg{deg}: max rel {mx:.2e} mean {mn:+.1e} p(hi)={p1:.7f} c={c}")


def fit_exp2_free_c0(lo, hi, deg, bf16_top):
    """2^f ~ c0 + ... + c_deg f^deg, relative minimax, top `bf16_top` coeffs rounded to BF16 with refits,
    then c0 scaled for zero mean relative error."""
    f = np.linspace(lo, hi, 40001)
    target = 2.0**f
    w = 1 / target
    fixed = {}
    while True:
        free = [k for k in range(deg + 1) if k not in fixed]
        resid = target - sum(fixed[k] * f**k for k in fixed)
        V = np.vstack([f**k for k in free]).T
        lw = np.ones_like(f) / len(f)
        for _ in range(300):
            sw = np.sqrt(lw) * w
            c, *_ = np.linalg.lstsq(V * sw[:, None], resid * sw, rcond=None)
            err = np.abs((V @ c - resid) * w)
            lw = lw * (err + 1e-30)
            lw /= lw.sum()
        coeffs = dict(zip(free, c))
        if len(fixed) == bf16_top:
            break
        top = max(free)
        fixed[top] = bf16_round(coeffs[top])
    coeffs.update(fixed)
    p = sum(coeffs[k] * f**k for k in coeffs)
    coeffs[0] -= np.mean(p / target - 1) / np.mean(1 / target)
    p = sum(coeffs[k] * f**k for k in coeffs)
    r = p / target - 1
    return [float(np.float32(coeffs[k])) for k in range(deg + 1)], np.abs(r).max(), np.mean(r)
