# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin (-p place_plugin, this dir on PYTHONPATH): t2p2_swiglu_pair_placement bench.

Replaces genify_matmul_program_descriptor.bank_layout (the core -> bank-class map of the R1b bank-local split) for
swiglu geometries (GMP_ALL=1: every geometry) with an explicit placement spec, without touching the op:

  GMP_PLACE   "u@r0,r1,.../xlo-xhi[+r,.../xlo-xhi];u@...": per pair class u (banks 2u = gate, 2u + 1 = up), its
              candidate cores in preference order: rows in the listed order, inside a row x in the listed range,
              nearest to the class's DRAM column in +x (NoC0 response direction) first.
              A segment suffix ":N" gives each of its cores exactly N units (assigned first); the rest of the
              class's units go in even runs over its unsuffixed cores.
              "auto:<name>": a named algorithmic placement (see AUTO below).
  GMP_FORCE_BANK=1  take the bank candidate whenever one exists (bypasses the split ranking).
  GMP_MAXW    max units (pairs) per core; a class uses ceil(units / MAXW) of its candidates (front first).
              0 / unset: all candidates, even runs.
Prints one GMPLAYOUT block per new layout (class digit per core, '.' idle; logical y rows, x columns)."""
import math
import os

import pytest

_printed = set()


def _parse(spec):
    """-> {u: [(rows, xlo, xhi, fixed units per core or 0)]}"""
    out = {}
    for part in filter(None, spec.split(";")):
        u, rest = part.split("@")
        cands = []
        for seg in rest.split("+"):
            seg, _, fixed = seg.partition(":")
            rows, xr = seg.split("/")
            lo, hi = (int(v) for v in xr.split("-"))
            cands.append(([int(r) for r in rows.split(",")], lo, hi, int(fixed or 0)))
        out[int(u)] = cands
    return out


def _auto(name, geom, gx, rows, U, homes):
    """Algorithmic specs. homes[u] = [(gate home x, y), (up home x, y)] (logical)."""
    ring = rows + 2  # NoC0 torus rows on this board: 2 non-worker rows between logical rows-1 and 0
    west = [x for x in range(gx) if x < 6]
    east = [x for x in range(gx) if x >= 6]

    def down(u):  # the home row downstream in +y (the up/gate home the other one reaches in fewer +y hops)
        (gx_, gy), (ux, uy) = homes[u]
        return uy if (uy - gy) % ring <= (gy - uy) % ring else gy

    spec = {}
    if name.startswith("down"):
        # each class on rows starting at its downstream home row, going +y, in its own DRAM column's half
        n = int(name[4:] or 3)
        for u in range(U):
            half = west if homes[u][0][0] < 6 else east
            r0 = down(u)
            spec[u] = [([(r0 + i) % rows for i in range(n)], min(half), max(half))]
    return spec


def _layout(geom, gx, rows, spec, maxw, pd):
    unit, U, sizes = pd.bank_classes(geom)
    homes = [[geom.bank_home[unit * u + i] for i in range(unit)] for u in range(U)]
    if spec == "auto:seam":
        import pair_layout

        return pair_layout.seam_layout(gx, rows, homes, sizes)
    if spec.startswith("auto:"):
        cands = _auto(spec[5:], geom, gx, rows, U, homes)
    else:
        cands = _parse(spec)
    out = {(x, y): (0, 0, 0) for y in range(rows) for x in range(gx)}
    taken = set()
    for u in range(U):
        hx = homes[u][0][0]
        order = []  # [(xy, fixed)]
        seen = set()
        for c in cands.get(u, []):
            rlist, lo, hi = c[:3]
            fixed = c[3] if len(c) > 3 else 0
            for r in rlist:
                for x in sorted(range(lo, hi + 1), key=lambda x: ((x - hx) % gx, x)):
                    if (x, r) in out and (x, r) not in taken and (x, r) not in seen:
                        order.append(((x, r), fixed))
                        seen.add((x, r))
        left = sizes[u]
        widths = {}
        for xy, f in order:  # fixed-width cores first
            if f and left:
                widths[xy] = min(f, left)
                left -= widths[xy]
        free = [xy for xy, f in order if not f]
        n = len(free) if not maxw else min(len(free), math.ceil(left / maxw))
        n = min(n, left)
        if left and n == 0:
            return None
        for xy, (_, w) in zip(free[:n], pd._even_runs(left, n)):
            widths[xy] = w
        t = 0
        for xy, _ in order:
            w = widths.get(xy, 0)
            if w:
                out[xy] = (u, t, w)
                taken.add(xy)
                t += w
        assert t == sizes[u], (u, t, sizes[u])
    return out


@pytest.fixture(autouse=True)
def _gmp_place(monkeypatch):
    from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

    if os.environ.get("GMP_FORCE_BANK") == "1":
        orig_split = pd.choose_split

        def choose_split(geom, gx, gy, budget):
            for grid in pd.bank_candidates(geom, gx, gy):
                blk = pd.choose_blocking(geom, grid, budget)
                if pd.operand_mode(geom, grid, blk) == 2:
                    return grid, blk
            return orig_split(geom, gx, gy, budget)

        monkeypatch.setattr(pd, "choose_split", choose_split)
    spec = os.environ.get("GMP_PLACE", "")
    if not spec:
        return
    maxw = int(os.environ.get("GMP_MAXW", "0") or 0)
    every = os.environ.get("GMP_ALL") == "1"
    orig = pd.bank_layout

    def bank_layout(geom, gx, rows, placement=None):
        if not (geom.swiglu or every):
            return orig(geom, gx, rows, placement)
        out = _layout(geom, gx, rows, spec, maxw, pd)
        key = (geom.tensor_m_tiles, geom.tensor_k_tiles, geom.tensor_n_tiles, gx, rows)
        if out is not None and key not in _printed:
            _printed.add(key)
            ws = [w for _, _, w in out.values()]
            print(f"GMPLAYOUT {key} spec={spec} maxw={maxw} max_units={max(ws)} working={sum(1 for w in ws if w)}")
            for y in range(rows):
                print(
                    f"GMPLAYOUT y{y}: "
                    + " ".join(f"{out[(x, y)][0]}{out[(x, y)][2]}" if out[(x, y)][2] else ".." for x in range(gx))
                )
        return out

    monkeypatch.setattr(pd, "bank_layout", bank_layout)


def pytest_configure(config):
    """GMP_GRAD=1: run the graduation (grad/genify_matmul_program_descriptor.py = HEAD + graduate_seam.patch): every
    top-level statement whose source differs from the live module's is executed in the live module's namespace."""
    if os.environ.get("GMP_GRAD") != "1":
        return
    import ast
    from pathlib import Path

    from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

    def nodes(src):
        out = {}
        for n in ast.parse(src).body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                out[n.name] = (n, ast.dump(n))
            elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                out[n.targets[0].id] = (n, ast.dump(n))
        return out

    live = nodes(Path(pd.__file__).read_text())
    grad = nodes((Path(__file__).resolve().parent / "grad" / "genify_matmul_program_descriptor.py").read_text())
    changed = [k for k in grad if k not in live or live[k][1] != grad[k][1]]
    # classes first (functions resolve them at call time anyway)
    for k in sorted(changed, key=lambda k: not isinstance(grad[k][0], ast.ClassDef)):
        exec(compile(ast.Module([grad[k][0]], []), str(pd.__file__), "exec"), pd.__dict__)
    print(f"GMPGRAD applied: {sorted(changed)}")
    orig = pd.choose_split

    def choose_split(geom, gx, gy, budget):
        grid, blk = orig(geom, gx, gy, budget)
        if grid.placement == "bank" and geom.swiglu:
            terms = [round(t / 1350, 1) for t in pd.split_cost_terms(geom, grid, blk)]
            pd.BANK_LOCAL_SWIGLU = False
            try:
                g2, b2 = orig(geom, gx, gy, budget)
            finally:
                pd.BANK_LOCAL_SWIGLU = True
            t2 = [round(t / 1350, 1) for t in pd.split_cost_terms(geom, g2, b2)]
            print(
                f"GMPBANK swiglu bank split m={geom.tensor_m_tiles} k={geom.tensor_k_tiles} n={geom.tensor_n_tiles} "
                f"terms={terms} {grid} {blk} || alt mode={pd.operand_mode(geom, g2, b2)} terms={t2} {g2} {b2}"
            )
        return grid, blk

    pd.choose_split = choose_split
