# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only prototype of the descriptor's round-0 reader rows (same code as op_copy's _br0_q_rows_model).
    python3 qsel2.py <round-0 bank per column> <sender rows> [spare: 0/1]"""
import sys

NOC1_EP_Y = {0: 1, 1: 10, 2: 4, 3: 7, 4: 1, 5: 10, 6: 4, 7: 7}


def q_rows(banks, sender_rows, rows, spare, ep_y=NOC1_EP_Y, y0=2):
    """Per column the reader's grid row: distinct rows, minimizing (max over columns of the climb's worst link sharing,
    plus one for a reader off the sender's row when spare members wait on the sender's relay), then the sum, the climb
    hops, the readers moved."""
    n = len(banks)
    half = 4

    def climb(b, r):
        y, side, links = ep_y[b], (0 if b < half else 1), []
        while y != r + y0:
            links.append((side, y))
            y = (y - 1) % 12
        return links

    climbs = {(k, r): climb(banks[k], r) for k in range(n) for r in rows}
    best = [None, None]
    order = sorted(range(n), key=lambda k: min(len(climbs[(k, r)]) for r in rows))

    def score(assign):
        use = {}
        for k, r in assign.items():
            for l in climbs[(k, r)]:
                use[l] = use.get(l, 0) + 1
        per = []
        for k, r in assign.items():
            s = max((use[l] for l in climbs[(k, r)]), default=1)
            per.append(s + (1 if spare and r != sender_rows[k] else 0))
        hops = sum(len(climbs[(k, r)]) for k, r in assign.items())
        moved = sum(1 for k, r in assign.items() if r != sender_rows[k])
        return (max(per), sum(per), hops, moved)

    def dfs(i, assign, used):
        if best[0] is not None and score(assign)[0] > best[0][0]:
            return
        if i == n:
            s = score(assign)
            if best[0] is None or s < best[0]:
                best[0], best[1] = s, dict(assign)
            return
        k = order[i]
        for r in sorted(rows, key=lambda r: (len(climbs[(k, r)]), r != sender_rows[k])):
            if r in used:
                continue
            assign[k] = r
            used.add(r)
            dfs(i + 1, assign, used)
            used.discard(r)
            del assign[k]

    dfs(0, {}, set())
    return [best[1][k] for k in range(n)], best[0]


if __name__ == "__main__":
    import time

    banks = [int(v) for v in sys.argv[1].split(",")]
    srows = [int(v) for v in sys.argv[2].split(",")]
    spare = len(sys.argv) > 3 and sys.argv[3] == "1"
    t = time.time()
    print(q_rows(banks, srows, list(range(10)), spare), f"{time.time() - t:.2f}s")
