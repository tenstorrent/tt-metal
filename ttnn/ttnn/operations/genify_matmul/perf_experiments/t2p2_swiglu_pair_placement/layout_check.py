"""Host-only check of place_plugin layouts (no ttnn import): prints the map + per-class core counts / max units."""
import sys, types
from types import SimpleNamespace as NS

sys.modules.setdefault("pytest", types.SimpleNamespace(fixture=lambda **k: (lambda f: f)))
import place_plugin as pp


def bank_classes(g):
    unit = 2 if g.swiglu else 1
    U = g.b_banks // unit
    n = g.tensor_n_tiles // unit
    return unit, U, [len(range(u, n, U)) for u in range(U)]


def _even_runs(n, cores):
    out, t = [], 0
    for i in range(cores):
        w = n // cores + (1 if i < n % cores else 0)
        out.append((t, w))
        t += w
    return out


pd = NS(bank_classes=bank_classes, _even_runs=_even_runs)
HOMES = ((0, 9), (0, 0), (0, 7), (0, 3), (6, 9), (6, 1), (6, 6), (6, 4))
nt = int(sys.argv[2]) if len(sys.argv) > 2 else 272
g = NS(swiglu=True, b_banks=8, tensor_n_tiles=nt, bank_home=HOMES)
out = pp._layout(g, 11, 10, sys.argv[1], int(sys.argv[3]) if len(sys.argv) > 3 else 0, pd)
for y in range(10):
    print(f"y{y}: " + " ".join(f"{out[(x, y)][0]}{out[(x, y)][2]}" if out[(x, y)][2] else ".." for x in range(11)))
from collections import Counter

print("max units", max(w for _, _, w in out.values()), "cores/class", Counter(u for u, _, w in out.values() if w))
