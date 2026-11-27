# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""t2p2: the swiglu pair-unit bank placement ("seam" placement), host-only (no ttnn import), shared by the plugin
and the graduation patch.

NoC0 delivers a DRAM read response +x along the bank's home row, then +y down the core's column. On this board the
+y torus seam (the last logical row -> the first) is slow (measured: a gate bank whose responses all cross it streams
at ~1/2 rate), while the +x seam is not. A pair class (gate bank 2u, up bank 2u + 1) on core row r crosses the seam for
every bank homed below r (hrow > r). Placement:
  1. units per core w: the smallest that fits (every core runs the grid's nominal max width, so w sets the K-walk);
  2. seam-free cells (r >= both home rows): the class with the fewest free seam-free cells claims first, its cells
     ordered by +x distance from its banks' DRAM column, then row; a row two classes need equally is split by
     that distance (a gate bank's own half);
  3. the rest crosses the seam for exactly one bank: greedy over (class, cell) by the column's seam load (units already
     crossing it there), then hops after the seam, then +x distance -- the seam traffic spreads over every column;
  4. then cells crossing for both banks, likewise.
Returns {(x, y): (u, t0, w)} or None."""


def _even_runs(n, cores):
    out, t = [], 0
    for i in range(cores):
        w = n // cores + (1 if i < n % cores else 0)
        out.append((t, w))
        t += w
    return out


def seam_layout(gx, rows, homes, sizes):
    """homes[u] = [(x, y) of each bank of class u] (logical NoC0 homes); sizes[u] = units of class u."""
    U = len(sizes)
    cells = [(x, y) for y in range(rows) for x in range(gx)]
    hx = [homes[u][0][0] for u in range(U)]

    def wraps(u, r):
        return sum(1 for _, hy in homes[u] if r < hy)

    def xpref(u, x):
        return (x - hx[u]) % gx

    def hops(u, r):  # +y hops after the seam to row r for the banks of u that cross it (largest)
        return max([r + rows - hy for _, hy in homes[u] if r < hy] or [0])

    total = sum(sizes)
    for w in range(max(1, -(-total // len(cells))), max(sizes) + 1):
        need = [-(-s // w) for s in sizes]
        if sum(need) > len(cells):
            continue
        owner = {}
        claimed = {u: [] for u in range(U)}

        def claim(u, xy):
            owner[xy] = u
            claimed[u].append(xy)
            need[u] -= 1

        # 2. seam-free cells
        while True:
            free0 = {u: [c for c in cells if c not in owner and wraps(u, c[1]) == 0] for u in range(U) if need[u] > 0}
            free0 = {u: f for u, f in free0.items() if f}
            if not free0:
                break
            scarce = min(len(f) for f in free0.values())
            group = sorted(u for u, f in free0.items() if len(f) == scarce)
            # contested cells go to the group member nearest in +x; a member keeps only what it needs
            mine = {u: [] for u in group}
            for c in sorted(set().union(*[free0[u] for u in group]), key=lambda c: (c[1], c[0])):
                cand = [u for u in group if c in free0[u]]
                mine[min(cand, key=lambda u: (xpref(u, c[0]), u))].append(c)
            for u in group:
                for c in sorted(mine[u], key=lambda c: (xpref(u, c[0]), c[1]))[: need[u]]:
                    claim(u, c)
            # a member short of its own cells takes the others' leftovers
            for u in group:
                rest = sorted([c for c in free0[u] if c not in owner], key=lambda c: (xpref(u, c[0]), c[1]))
                for c in rest[: max(0, need[u])]:
                    claim(u, c)
        # 3./4. seam cells: one crossing bank, then two
        load = [0] * gx
        for k in (1, 2):
            while any(n > 0 for n in need):
                best = None
                for u in range(U):
                    if need[u] <= 0:
                        continue
                    for c in cells:
                        if c in owner or wraps(u, c[1]) != k:
                            continue
                        key = (load[c[0]] + w * k, hops(u, c[1]), -need[u], xpref(u, c[0]), c[1], u)
                        if best is None or key < best[0]:
                            best = (key, u, c)
                if best is None:
                    break
                _, u, c = best
                claim(u, c)
                load[c[0]] += w * k
        if any(n > 0 for n in need):
            continue
        out = {c: (0, 0, 0) for c in cells}
        for u in range(U):
            for xy, (t0, wu) in zip(claimed[u], _even_runs(sizes[u], len(claimed[u]))):
                out[xy] = (u, t0, wu)
        return out
    return None


if __name__ == "__main__":
    import sys

    HOMES = ((0, 9), (0, 0), (0, 7), (0, 3), (6, 9), (6, 1), (6, 6), (6, 4))
    nt = int(sys.argv[1]) if len(sys.argv) > 1 else 272
    homes = [[HOMES[2 * u], HOMES[2 * u + 1]] for u in range(4)]
    npair = nt // 2
    sizes = [len(range(u, npair, 4)) for u in range(4)]
    out = seam_layout(11, 10, homes, sizes)
    for y in range(10):
        print(f"y{y}: " + " ".join(f"{out[(x, y)][0]}{out[(x, y)][2]}" if out[(x, y)][2] else ".." for x in range(11)))
    print("sizes", sizes, "max", max(w for _, _, w in out.values()))
