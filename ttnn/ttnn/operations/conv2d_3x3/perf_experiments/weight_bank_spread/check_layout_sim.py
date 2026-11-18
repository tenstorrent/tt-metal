# Host-only check (no ttnn): the writer copy's per-round read plans and weight_page() vs prepare's permutation.
# python3 check_layout_sim.py op_copy/conv2d_3x3.py
# host-only check: kernel read plans (layouts 1, 2) and weight_page() vs the host permutation
import ast, sys

src = open(sys.argv[1]).read()
mod = ast.parse(src)
fn = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "weight_page_permutation"][0]
ns = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "x", "exec"), ns)
perm_fn = ns["weight_page_permutation"]
NB = 8


def kernel_weight_page(layout, col, K, T, out_n, k, n):
    col_page_base = col * K * T
    if layout == 0:
        return (col * K + k) * T + n
    n0 = (n // out_n) * out_n
    valid = min(out_n, T - n0)
    run = col_page_base + n0 * K
    j = k * valid + (n - n0)
    if layout == 1:
        return run + j
    L = K * valid
    q, r = divmod(L, NB)
    if j < r * (q + 1):
        return run + j // (q + 1) + NB * (j % (q + 1))
    jj = j - r * (q + 1)
    return run + r + jj // q + NB * (jj % q)


def kernel_round(layout, col, K, T, out_n, y):
    # returns dict l1_slot -> page
    res = {}
    n0 = y * out_n
    valid = min(out_n, T - n0)
    base = col * K * T
    if layout == 1:
        page = base + n0 * K
        for k in range(K):
            for n in range(valid):
                res[k * out_n + n] = page
                page += 1
    else:
        run = base + n0 * K
        L = K * valid
        q, r = divmod(L, NB)
        j = 0
        for m in range(min(L, NB)):
            cnt = q + (1 if m < r else 0)
            t = 0
            while cnt > 0:
                k, n = divmod(j, valid)
                seg = min(valid - n, cnt)
                for s in range(seg):
                    res[k * out_n + n + s] = run + m + NB * (t + s)
                t += seg
                j += seg
                cnt -= seg
    return res


bad = 0
for S, K, T in [
    (11, 54, 8),
    (8, 18, 16),
    (8, 9, 8),
    (3, 9, 8),
    (4, 27, 5),
    (2, 9, 3),
    (5, 6, 7),
    (11, 27, 8),
    (2, 3, 1),
]:
    out_n = -(-T // S)
    for layout in (1, 2):
        perm = perm_fn(S, K, T, layout, NB)
        inv = {v: p for p, v in enumerate(perm)}
        for col in range(S):
            for y in range(S):
                if y * out_n >= T:
                    continue
                n0 = y * out_n
                for slot, page in kernel_round(layout, col, K, T, out_n, y).items():
                    k, dn = divmod(slot, out_n)
                    want = (col * K + k) * T + n0 + dn
                    if inv[page] != want:
                        bad += 1
            for k in range(K):
                for n in range(T):
                    if inv[kernel_weight_page(layout, col, K, T, out_n, k, n)] != (col * K + k) * T + n:
                        bad += 1
        # banks per round
        if (S, K, T) in [(11, 54, 8), (8, 18, 16)] and layout == 2:
            r = kernel_round(layout, 0, K, T, out_n, 0)
            print(S, K, T, "chunk round0 banks", sorted({p % NB for p in r.values()}))
print("mismatches", bad)
