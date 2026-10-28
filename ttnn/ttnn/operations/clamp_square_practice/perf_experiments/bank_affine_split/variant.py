# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""bank_affine_split: each core owns pages of the DRAM bank(s) NoC-nearest to it.

Starts from the op's own create_program_descriptor (CBs, block size, compute kernel, core set and
per-core tile counts untouched) and swaps only the DM kernels + their RT args.

Distance model (host side, from the real device, nothing fitted to this board):
  * get_optimal_dram_bank_to_logical_worker_assignment(NOC_k) gives, per bank, the logical worker
    placed immediately downstream ("to the right") of that bank's NoC_k endpoint (metal derives
    it from physical NOC0 coordinates + harvesting). Call it anchor_k[b].
  * NoC0 routes +x/+y, NoC1 routes -x/-y (unidirectional rings). The data-carrying leg is
    bank->core for a NoC0 read and core->bank for a NoC1 write; on both NoCs that leg's length
    is ~ ((cx - ax) mod GX) + ((cy - ay) mod GY) with a = anchor of the bank on that NoC.
  * cost(core, bank) = w_r * d(anchor_0[bank], core) + w_w * d(anchor_1[bank], core).
Assignment = min-cost transportation: bank b supplies its n_b pages (pages b, b+NB, ...), core c
demands exactly its baseline tile count (so per-core work == op baseline). Each bank's j-range is
then cut into contiguous runs, one per receiving core -> per-core list of (bank, j0, cnt).

Bank groups: the banks are partitioned into ordered groups from the device anchors
  bank  one bank per group           (pure bank affinity: consecutive addresses in one bank)
  pair  2 banks of one DRAM column, adjacent endpoint rows
  col   all banks of one DRAM column (anchor x)
  all   one group = every bank       (== the op's contiguous split; kernel-overhead sanity)
A core walks its group's pages in natural order, round-robin over the group's banks.
Modes "<grouping>_<how>": how = near (min-cost transport, balanced) | far (max-cost control) |
  nearrd (read leg only) | nearest (argmin group per core, UNBALANCED per-core counts).
"""
from pathlib import Path

import numpy as np
import ttnn
from ttnn.operations.clamp_square_practice import clamp_square_practice_program_descriptor as pd

KDIR = Path(__file__).parent / "kernels"


def _anchors(device, noc):
    a = device.get_optimal_dram_bank_to_logical_worker_assignment(noc)
    if isinstance(a, dict):
        return [(a[k].x, a[k].y) for k in sorted(a)]
    return [(c.x, c.y) for c in a]


def cost_matrix(device, cores, w_r=1.0, w_w=1.0):
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    a0 = _anchors(device, ttnn.NOC.NOC_0)
    a1 = _anchors(device, ttnn.NOC.NOC_1)

    def d(a, c):
        return (c[0] - a[0]) % gx + (c[1] - a[1]) % gy

    return np.array([[w_r * d(a0[b], c) + w_w * d(a1[b], c) for b in range(len(a0))] for c in cores], float)


def _transport(cost, caps, supply):
    """Exact min-cost transportation (integral vertex). caps: per-core demand, supply: per-bank pages."""
    from scipy.optimize import linprog

    nc, nb = cost.shape
    A = np.zeros((nc + nb, nc * nb))
    for c in range(nc):
        A[c, c * nb : (c + 1) * nb] = 1
    for b in range(nb):
        A[nc + b, b::nb] = 1
    res = linprog(cost.ravel(), A_eq=A, b_eq=np.concatenate([caps, supply]), bounds=(0, None), method="highs-ds")
    assert res.status == 0, res.message
    x = np.rint(res.x).astype(int).reshape(nc, nb)
    assert np.abs(res.x.reshape(nc, nb) - x).max() < 1e-6, "non-integral transportation vertex"
    return x


def bank_groups(device, grouping):
    """Partition banks into ordered groups, derived from the device's per-bank anchors."""
    nb = device.dram_grid_size().x
    a0 = _anchors(device, ttnn.NOC.NOC_0)
    if grouping == "bank":
        return [[b] for b in range(nb)]
    if grouping == "all":
        return [list(range(nb))]
    cols = {}
    for b in range(nb):
        cols.setdefault(a0[b][0], []).append(b)
    if grouping == "col":
        return [sorted(v) for _, v in sorted(cols.items())]
    if grouping == "pair":  # neighbours in the same DRAM column, by endpoint row
        out = []
        for _, v in sorted(cols.items()):
            v = sorted(v, key=lambda b: a0[b][1])
            out += [sorted(v[i : i + 2]) for i in range(0, len(v), 2)]
        return out
    raise ValueError(grouping)


def group_supply(groups, n_pages, nb):
    full, rem = divmod(n_pages, nb)
    return np.array([full * len(G) + sum(1 for b in G if b < rem) for G in groups])


def _transport_greedy(cost, caps, supply):
    """Dependency-free transportation (the form to graduate): cores in decreasing regret order
    (2nd-cheapest minus cheapest group; stable on core index) fill from their cheapest group with
    supply left. Exact coverage by construction; a core spans at most #groups segments."""
    nc, ng = cost.shape
    left = np.array(supply, dtype=int).copy()
    x = np.zeros((nc, ng), dtype=int)
    srt = np.sort(cost, axis=1)
    regret = srt[:, 1] - srt[:, 0] if ng > 1 else np.zeros(nc)
    for c in sorted(range(nc), key=lambda c: (-regret[c], c)):
        need = int(caps[c])
        for gi in sorted(range(ng), key=lambda g: (cost[c, g], g)):
            take = min(need, left[gi])
            x[c, gi] += take
            left[gi] -= take
            need -= take
            if need == 0:
                break
        assert need == 0
    return x


def assign(device, cores, caps, n_pages, mode="bank_near", lp=False):
    """-> per-core list of segments (G, q0, cnt) and per-core tile counts.

    mode = "<grouping>_<near|far|nearrd>" with grouping in bank|pair|col|all, or "bank_nearest"."""
    nb = device.dram_grid_size().x
    grouping, how = mode.split("_")
    groups = bank_groups(device, grouping)
    supply = group_supply(groups, n_pages, nb)
    caps = np.array(caps)
    w = (1, 0) if how == "nearrd" else (1, 1)
    cb = cost_matrix(device, cores, *w)
    cost = np.stack([cb[:, G].mean(axis=1) for G in groups], axis=1)
    if how == "nearest":
        # argmin group per core; ties go to the tied group with the fewest members so far.
        load = np.zeros(len(groups), int)
        own = np.zeros(len(cores), int)
        for c in range(len(cores)):
            tied = np.flatnonzero(cost[c] == cost[c].min())
            own[c] = tied[np.argmin(load[tied])]
            load[own[c]] += 1
        x = np.zeros_like(cost, dtype=int)
        for gi in range(len(groups)):
            members = np.flatnonzero(own == gi)
            if len(members) == 0:  # no core has it as nearest: give it to its cheapest core
                members = np.array([cost[:, gi].argmin()])
            q, r = divmod(supply[gi], len(members))
            for i, c in enumerate(members):
                x[c, gi] = q + (1 if i < r else 0)
    else:
        solver = _transport if lp else _transport_greedy
        x = solver(-cost if how == "far" else cost, caps, supply)
    segs = [[] for _ in cores]
    for gi, G in enumerate(groups):
        q = 0
        for c in range(len(cores)):
            if x[c, gi]:
                segs[c].append((G, q, int(x[c, gi])))
                q += int(x[c, gi])
        assert q == supply[gi]
    counts = [int(v) for v in x.sum(axis=1)]
    pages = sorted((q // len(G)) * nb + G[q % len(G)] for s in segs for (G, q0, n) in s for q in range(q0, q0 + n))
    assert pages == list(range(n_pages)), "page assignment must be a permutation"
    return segs, counts


def build(x, out, min_val, max_val, mode="bank_near", skip_compute=False):
    desc = pd.create_program_descriptor(x, out, min_val, max_val)
    rk, wk, ck = desc.kernels
    device = x.device()
    grid = device.compute_with_storage_grid_size()
    n_pages = x.buffer_num_pages()
    nb = device.dram_grid_size().x
    (_, all_cores, g1, _, t1, t2) = ttnn.split_work_to_cores(grid, n_pages, row_wise=True)
    g1s = set((c.x, c.y) for c in ttnn.corerange_to_cores(g1, row_wise=True))
    cores = ttnn.corerange_to_cores(all_cores, row_wise=True)
    caps = [t1 if (c.x, c.y) in g1s else t2 for c in cores]
    segs, counts = assign(device, [(c.x, c.y) for c in cores], caps, n_pages, mode)
    groups = bank_groups(device, mode.split("_")[0])
    nslot, gmax = len(groups), max(len(G) for G in groups)
    assert all(len(s) <= nslot for s in segs)

    rrt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    in_addr, out_addr = x.buffer_address(), out.buffer_address()
    crt_old = ck.runtime_args
    crt = ttnn.RuntimeArgs()
    for c, s, n in zip(cores, segs, counts):
        # Fixed-length RT args: NSLOT = #groups slots of (g, row0, k0, cnt, bank[0..GMAX-1]); a core
        # holds at most one segment per group. Constant length keeps the program-cache RT-arg
        # override valid (a cache hit cannot grow a core's args).
        flat = []
        for G, q0, cnt in s:
            flat += [len(G), q0 // len(G), q0 % len(G), cnt] + list(G) + [0] * (gmax - len(G))
        flat += [0] * ((4 + gmax) * (nslot - len(s)))
        rrt[c.x][c.y] = [in_addr, n, len(s)] + flat
        wrt[c.x][c.y] = [out_addr, n, len(s)] + flat
        old = list(crt_old[c.x][c.y])
        crt[c.x][c.y] = [n] + old[1:]
    rk.runtime_args, wk.runtime_args, ck.runtime_args = rrt, wrt, crt

    rct, wct = list(rk.compile_time_args), list(wk.compile_time_args)
    rk.compile_time_args = rct[:3] + [nb, gmax] + rct[3:]
    wk.compile_time_args = wct[:3] + [nb, gmax] + wct[3:]
    rk.kernel_source = str(KDIR / "reader.cpp")
    wk.kernel_source = str(KDIR / "writer.cpp")
    if skip_compute:
        ck.defines = [("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1")]
    desc.kernels = [rk, wk, ck]
    return desc


def build_baseline(x, out, min_val, max_val, skip_compute=False):
    desc = pd.create_program_descriptor(x, out, min_val, max_val)
    if skip_compute:
        rk, wk, ck = desc.kernels
        ck.defines = [("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1")]
        desc.kernels = [rk, wk, ck]
    return desc
