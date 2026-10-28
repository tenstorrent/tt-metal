# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""noc_balance experiment: reshape the op's DRAM/NoC traffic without touching compute.

Starts from the op's own create_program_descriptor (CBs, block size, compute kernel untouched),
then swaps reader/writer for kernels/reader.cpp + kernels/writer.cpp and rewrites their RT args.

Knobs:
  rd_noc / wr_noc : 0 home NoC (baseline), 1 alternate per page, 2 alternate per block,
                    3 other NoC, 4 per-core NoC from `core_noc(x, y, role)`, 5 per-bank mask.
  order           : "contig" (op baseline), "rot" (core k's walk starts on bank k % NB),
                    "strided" (core k takes pages k, k+N, k+2N ...).
  skip_compute    : DM-only bench (CKL_ELTWISE_CHAIN_SKIP_COMPUTE=1).
Any non-zero NoC mode puts BOTH DM kernels in DM_DYNAMIC_NOC (needed to issue on the non-home NoC).
"""
from pathlib import Path

import ttnn
from ttnn.operations.clamp_square_practice import clamp_square_practice_program_descriptor as pd

KDIR = Path(__file__).parent / "kernels"


def build(
    x,
    out,
    min_val,
    max_val,
    rd_noc=0,
    wr_noc=0,
    order="contig",
    skip_compute=False,
    dynamic=None,
    core_noc=None,
    bank_mask=None,
    ablate="",
    split_writer=None,
):
    desc = pd.create_program_descriptor(x, out, min_val, max_val)
    rk, wk, ck = desc.kernels
    device = x.device()
    grid = device.compute_with_storage_grid_size()
    n_pages = x.buffer_num_pages()
    nb = device.dram_grid_size().x
    (n_cores, all_cores, g1, g2, t1, t2) = ttnn.split_work_to_cores(grid, n_pages, row_wise=True)
    g1s = set((c.x, c.y) for c in ttnn.corerange_to_cores(g1, row_wise=True))
    in_addr, out_addr = x.buffer_address(), out.buffer_address()

    rrt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    covered = []
    start = 0
    for k, c in enumerate(ttnn.corerange_to_cores(all_cores, row_wise=True)):
        cnt = t1 if (c.x, c.y) in g1s else t2
        if order == "strided":
            s, stride, rot = k, n_cores, 0
        elif order == "rot":
            s, stride = start, 1
            rot = ((k - start) % nb) % cnt if cnt else 0  # first page's bank = k % nb
        else:
            s, stride, rot = start, 1, 0
        covered.extend(s + ((i + rot) % cnt) * stride for i in range(cnt))

        def sel(mode, role):
            if mode == 4:
                return core_noc(c.x, c.y, role)
            if mode == 5:
                return bank_mask(c, role)
            return 0

        rsel, wsel = sel(rd_noc, "r"), sel(wr_noc, "w")
        rrt[c.x][c.y] = [in_addr, s, cnt, stride, rot, rsel]
        wrt[c.x][c.y] = [out_addr, s, cnt, stride, rot, wsel]
        start += cnt
    assert sorted(covered) == list(range(n_pages)), "page assignment must be a permutation"

    if dynamic is None:
        dynamic = rd_noc != 0 or wr_noc != 0
    mode = ttnn.NOC_MODE.DM_DYNAMIC_NOC if dynamic else ttnn.NOC_MODE.DM_DEDICATED_NOC
    rk.kernel_source = str(KDIR / "reader.cpp")
    wk.kernel_source = str(KDIR / "writer.cpp")
    rk.runtime_args = rrt
    wk.runtime_args = wrt
    rk.config = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0, noc_mode=mode
    )
    wk.config = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1, noc_mode=mode
    )
    # NB: descriptor .defines is a copy-on-read property — build the list, then assign once.
    rk.defines = [("RD_NOC_MODE", str(rd_noc)), ("NB_NUM_BANKS", str(nb))] + (
        [("ABLATE_READ", "1")] if "r" in ablate else []
    )
    wk.defines = [("WR_NOC_MODE", str(wr_noc)), ("NB_NUM_BANKS", str(nb))] + (
        [("ABLATE_WRITE", "1")] if "w" in ablate else []
    )
    if skip_compute:
        ck.defines = [("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1")]
    kernels = [rk, wk, ck]
    if split_writer is not None:
        # Dedicated-NoC per-core writer NoC: TWO writer kernels on disjoint core sets, one with
        # noc=NOC_1 (home) and one with noc=NOC_0. No DM_DYNAMIC_NOC needed (reader owns reads on
        # NoC0, writer owns writes on NoC0 — different cmd bufs and HW counters).
        assert not dynamic and wr_noc == 0
        cores = ttnn.corerange_to_cores(all_cores, row_wise=True)
        sets = {0: [], 1: []}
        for c in cores:
            sets[split_writer(c.x, c.y, "w")].append(c)
        kernels = [rk, ck]
        for noc_id, cs in sets.items():
            if not cs:
                continue
            k = ttnn.KernelDescriptor(
                kernel_source=wk.kernel_source,
                core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs]),
                compile_time_args=list(wk.compile_time_args),
                defines=wk.defines,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0,
                    noc=ttnn.NOC.NOC_0 if noc_id == 0 else ttnn.NOC.NOC_1,
                    noc_mode=ttnn.NOC_MODE.DM_DEDICATED_NOC,
                ),
            )
            ra = ttnn.RuntimeArgs()
            for c in cs:
                ra[c.x][c.y] = list(wrt[c.x][c.y])
            k.runtime_args = ra
            kernels.append(k)
    desc.kernels = kernels
    return desc


def build_baseline(x, out, min_val, max_val, skip_compute=False):
    """The op's real descriptor + kernels, untouched (honest baseline)."""
    desc = pd.create_program_descriptor(x, out, min_val, max_val)
    if skip_compute:
        rk, wk, ck = desc.kernels
        ck.defines = [("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1")]
        desc.kernels = [rk, wk, ck]
    return desc


# ---- Route-aware per-(core, bank) NoC choice (Blackhole p150b, this box) ----
# Physical NoC0 coords. DRAM worker endpoints per bank from soc_descriptors/blackhole_140_arch.yaml
# (dram_views[b].worker_endpoint = [noc0 subchannel, noc1 subchannel]). Tensix physical columns of this
# board (harvested 10 & 16) as reported by the device profiler; logical y -> physical y + 2.
BH_W, BH_H = 17, 12
BH_DRAM_EP = {  # bank: (noc0 endpoint, noc1 endpoint)
    0: ((0, 11), (0, 1)),
    1: ((0, 2), (0, 10)),
    2: ((0, 9), (0, 4)),
    3: ((0, 5), (0, 7)),
    4: ((9, 11), (9, 1)),
    5: ((9, 3), (9, 10)),
    6: ((9, 8), (9, 4)),
    7: ((9, 6), (9, 7)),
}
BH_PHYS_X = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15]


def _hops(src, dst, noc):
    """Dimension-ordered hop count on the torus: NoC0 travels +x/+y, NoC1 -x/-y."""
    (sx, sy), (dx, dy) = src, dst
    if noc == 0:
        return (dx - sx) % BH_W + (dy - sy) % BH_H
    return (sx - dx) % BH_W + (sy - dy) % BH_H


def bank_mask_min_hops(c, role, home=None):
    """Bit b set -> use NoC1 for pages on bank b. Picks the NoC whose DATA path is shorter
    (reads: bank -> core; writes: core -> bank). Ties go to the role's home NoC."""
    core = (BH_PHYS_X[c.x], c.y + 2)
    home = (0 if role == "r" else 1) if home is None else home
    m = 0
    for b, (e0, e1) in BH_DRAM_EP.items():
        if role == "r":
            h0, h1 = _hops(e0, core, 0), _hops(e1, core, 1)
        else:
            h0, h1 = _hops(core, e0, 0), _hops(core, e1, 1)
        use1 = h1 < h0 or (h1 == h0 and home == 1)
        m |= int(use1) << b
    return m
