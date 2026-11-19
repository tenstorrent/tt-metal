# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: NoC1 link-sharing model of concurrent DRAM read responses on Blackhole p150b (11x10 worker grid).
A NoC1 response from DRAM endpoint (xd, yd) to core (xc, yc) goes north (y decreasing, mod 12) along x = xd to yc,
then west (x decreasing, mod 17) along y = yc to xc ("yx"), or the other order ("xy"). Prints, per flow, the most
flows sharing one of its links.
    python3 noc1_model.py <yx|xy> <q rows comma list | s> [round-1 flows: 1]"""
import sys

XS = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]  # logical x -> physical x
NOC1_EP = {0: (0, 1), 1: (0, 10), 2: (0, 4), 3: (0, 7), 4: (9, 1), 5: (9, 10), 6: (9, 4), 7: (9, 7)}


def path(d, c, order):
    (xd, yd), (xc, yc) = d, c
    links = []
    x, y = xd, yd

    def north():
        nonlocal y
        while y != yc:
            ny = (y - 1) % 12
            links.append(("v", x, y, ny))
            y = ny

    def west():
        nonlocal x
        while x != xc:
            nx = (x - 1) % 17
            links.append(("h", y, x, nx))
            x = nx

    if order == "yx":
        north()
        west()
    else:
        west()
        north()
    return links


def flows_for(rows, round1=False, k_cols=8):
    out = []
    for k in range(k_cols):
        out.append((f"c{k}r0", NOC1_EP[(k - 1) % 8], (XS[k], rows[k] + 2)))
        if round1:
            out.append((f"c{k}r1", NOC1_EP[(k - 2) % 8], (XS[k], k + 2)))
    return out


def share(flows, order):
    use = {}
    paths = {}
    for n, d, c in flows:
        p = path(d, c, order)
        paths[n] = p
        for l in p:
            use[l] = use.get(l, 0) + 1
    return {n: (max((use[l] for l in p), default=0), len(p)) for n, p in paths.items()}


if __name__ == "__main__":
    order = sys.argv[1]
    rows = list(range(8)) if sys.argv[2] == "s" else [int(v) for v in sys.argv[2].split(",")]
    r1 = len(sys.argv) > 3 and sys.argv[3] == "1"
    res = share(flows_for(rows, r1), order)
    print(" ".join(f"{n}:{s}/{h}" for n, (s, h) in res.items() if n.endswith("r0")))
