"""Device-agnostic form of the P2 policy (what the graduation patch computes on the host): everything in the virtual
NoC coordinates the host has (worker_core_from_logical_core), DRAM endpoints located through the optimal NoC0 / NoC1
DRAM-bank workers (the endpoint is the column just west of its worker, in the worker's row), the NoC torus of the arch.
Per core: hops of the NoC1 write (north then west) to each endpoint, of the NoC0 write (east then south) to the NoC0
endpoint; all tiles go on the NoC with fewer hops (ties: the TAIL_NC_PCT split), the writer to its nearer endpoint."""


def tail_policy(cores, w0, w1, torus, nc_pct=36):
    """cores {key: ((vx, vy), u, w)}; w0 / w1: per bank the NoC0 / NoC1 optimal worker (vx, vy); torus (NX, NY).
    Returns {key: (reader_tiles, writer_endpoint_noc)}."""
    NX, NY = torus

    def h1(c, wk):  # NoC1: -y first, then -x, to the DRAM column west of worker wk (x - 0.5)
        return (c[1] - wk[1]) % NY + ((c[0] - wk[0]) % NX + 0.5)

    def h0(c, wk):  # NoC0: +x first, then +y
        return ((wk[0] - c[0]) % NX - 0.5) % NX + (wk[1] - c[1]) % NY

    out = {}
    for key, (c, u, w) in cores.items():
        d1 = {0: h1(c, w0[u]), 1: h1(c, w1[u])}
        wr = min(d1, key=lambda e: (d1[e], e != 1))
        H1, H0 = d1[wr], h0(c, w0[u])
        nc = w if H0 < H1 else (0 if H1 < H0 else (w * nc_pct) // 100)
        out[key] = (nc, wr)
    return out


if __name__ == "__main__":
    import sys

    sys.path.insert(0, sys.path[0])
    from noc_model import layout
    from policy import policy

    VX = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]
    W0 = [(VX[x], y + 2) for x, y in ((0, 9), (0, 0), (0, 7), (0, 3), (6, 9), (6, 1), (6, 6), (6, 4))]
    W1 = [(VX[x], y + 2) for x, y in ((0, 0), (0, 8), (0, 2), (0, 5), (6, 0), (6, 8), (6, 2), (6, 5))]
    lay = layout()
    cores = {xy: ((VX[xy[0]], xy[1] + 2), u, w) for xy, (u, t0, w) in lay.items() if w}
    pv = tail_policy(cores, W0, W1, (17, 12))
    pp = policy("P2_r0")
    diff = [(xy, pv[xy], (pp[xy]["nc"], pp[xy]["wr_ep"])) for xy in pv if pv[xy] != (pp[xy]["nc"], pp[xy]["wr_ep"])]
    print("cores", len(pv), "differ", len(diff))
    for d in diff:
        print(" ", d)
