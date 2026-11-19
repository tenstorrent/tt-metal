# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the round-0 reader row selection (Blackhole NoC1 vertical-sharing model), standalone for checking.
A NoC1 DRAM read response climbs north (y decreasing, mod 12) along the bank's DRAM column (x 0: banks 0-3, x 9: banks
4-7) from the bank's NoC1 endpoint row to the reader's row, then runs west along the reader's row. Readers on distinct
rows never share the row part; the column part is shared by every read of that DRAM column whose climb overlaps.
    python3 qsel.py <round-0 bank per column, comma list> [reader rows to evaluate, comma list]"""
import itertools
import sys

NOC1_EP_Y = {0: 1, 1: 10, 2: 4, 3: 7, 4: 1, 5: 10, 6: 4, 7: 7}  # non-relocated DRAM endpoints


def climb(bank, y):
    yd = NOC1_EP_Y[bank]
    side = 0 if bank < 4 else 1
    links = []
    while yd != y:
        links.append((side, yd))
        yd = (yd - 1) % 12
    return links


def cost(banks, ys):
    use = {}
    paths = [climb(b, y) for b, y in zip(banks, ys)]
    for p in paths:
        for l in p:
            use[l] = use.get(l, 0) + 1
    per = [max((use[l] for l in p), default=1) for p in paths]
    return per


def select(banks, rows_y):
    """Greedy: columns in order of their best achievable climb; each takes the free row with the least
    (max sharing after adding it, climb length)."""
    order = sorted(range(len(banks)), key=lambda k: min(len(climb(banks[k], y)) for y in rows_y))
    chosen = {}
    for k in order:
        best = None
        for y in rows_y:
            if y in chosen.values():
                continue
            trial = dict(chosen)
            trial[k] = y
            ks = sorted(trial)
            c = cost([banks[i] for i in ks], [trial[i] for i in ks])
            key = (max(c), sum(c), len(climb(banks[k], y)))
            if best is None or key < best[0]:
                best = (key, y)
        chosen[k] = best[1]
    return [chosen[k] for k in range(len(banks))]


if __name__ == "__main__":
    banks = [int(v) for v in sys.argv[1].split(",")]
    rows_y = list(range(2, 12))
    if len(sys.argv) > 2:
        ys = [int(v) + 2 for v in sys.argv[2].split(",")]
        print("eval", [y - 2 for y in ys], cost(banks, ys), [len(climb(b, y)) for b, y in zip(banks, ys)])
    sel = select(banks, rows_y)
    print("select rows", [y - 2 for y in sel], cost(banks, sel), [len(climb(b, y)) for b, y in zip(banks, sel)])
