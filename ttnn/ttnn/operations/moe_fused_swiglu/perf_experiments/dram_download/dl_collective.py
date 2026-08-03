# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Collective (L1->L1) traffic floor: the op's x-multicast, column reduce-scatter and h all-gather,
with NO compute and NO round rendezvous.

Byte volumes per M-block at count 256 / 88 cores — the collectives move 3.1x the DRAM bytes:

    x row-multicast   8 rows x 30 464 B x 10 dests x 8 grid rows =  19.50 MB
    reduce-scatter    8 contributors x 6 528 B x 2 x 88 cores    =   9.19 MB
    h all-gather      11 rounds x 52 224 B x 88 dests            =  50.55 MB
    total                                                          79.24 MB   (DRAM: 25.23 MB)

Phases compose so the cost can be attributed cumulatively (none -> x -> +reduce -> +h).
"""

from pathlib import Path

import ttnn

from ttnn.operations.moe_fused_swiglu.perf_experiments.dram_download import dl_bench as base

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE, BFP8_TILE, HIDDEN = 32, 1088, 2048


#: Which RISC carries each collective. The op's shipped assignment: the h all-gather and the x
#: row-multicast both ride the READER (NCRISC / NOC_0 — `HSEND="reader"`, and `HSEND=writer` measured a
#: null at +5-7 %), and the column reduce-scatter is SPLIT across both (`SCATTER_NOC="split"`: gate on
#: the writer, up on the reader).
#:
#: This matters because the first version of this bench ran EVERY phase on BOTH RISCs, so each root
#: multicast h twice and the h figure was ~2x the real traffic.
OP_ASSIGN = {"x": (1, 0), "reduce": (1, 1), "h": (1, 0)}  # (on_reader, on_writer)


def build(device, p, *, phases, m_eff, assign=None, posted=0, single_root=False):
    hgroups, kgroups = p["hgroups"], p["kgroups"]
    hn_sizes, _ = p["hn"]
    kr_sizes, _ = p["kr"]
    hn_max, kr_max = max(hn_sizes), max(kr_sizes)
    num_cores = hgroups * kgroups

    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(hgroups - 1, kgroups - 1))])
    assign = assign or OP_ASSIGN

    def flags(risc):  # risc 0 = reader/NOC_0, 1 = writer/NOC_1
        return [1 if (ph in phases and assign[ph][risc]) else 0 for ph in ("x", "reduce", "h")]

    block = m_eff * hn_max * BFP8_TILE
    # Source holds the largest payload any phase sends; landing holds what the widest fan-in delivers.
    src_bytes = max(block, m_eff * kr_max * BFP8_TILE)
    land_bytes = max(2 * kgroups * (block // kgroups), block, m_eff * kr_max * BFP8_TILE)

    def cb(index, total, page):
        return ttnn.CBDescriptor(
            total_size=total,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat8_b, page_size=page)
            ],
        )

    # One private (src, land) pair per RISC — a CB has one producer, and both RISCs write here.
    cbs = [
        cb(0, src_bytes, BFP8_TILE),
        cb(1, land_bytes, BFP8_TILE),
        cb(2, src_bytes, BFP8_TILE),
        cb(3, land_bytes, BFP8_TILE),
    ]

    def ct_for(risc, src_idx, land_idx):
        return flags(risc) + [hgroups, kgroups, BFP8_TILE, num_cores, src_idx, land_idx, posted]

    # Physical NoC coords: the multicast rectangles and the per-column unicast targets.
    def phys(cx, cy):
        c = device.worker_core_from_logical_core(ttnn.CoreCoord(cx, cy))
        return int(c.x), int(c.y)

    a0 = phys(0, 0)
    a1 = phys(hgroups - 1, kgroups - 1)
    inj = base._injectors(p)

    # MULTICAST RECTANGLE CORNER ORDER IS NoC-DEPENDENT. NOC_1's traversal is the reverse of NOC_0's, so
    # the writer must be handed FAR->NEAR and the reader NEAR->FAR. Passing near->far to both gave BRISC
    # a reversed rectangle whose destinations never acked: every injector's BRISC hung in
    # `noc_async_write_barrier` (triage: dl_collective.cpp:78 on logical (1,0)..(7,0), NCRISC parked in
    # wait_for_brisc_notification). The op does the same thing at descriptor.py:1538 for the same reason.
    rts = []
    for risc in range(2):  # 0 = reader/NOC_0, 1 = writer/NOC_1
        rt = ttnn.RuntimeArgs()
        for x in range(hgroups):
            for y in range(kgroups):
                near_row, far_row = phys(0, y), phys(hgroups - 1, y)
                rowc = (far_row, near_row) if risc else (near_row, far_row)
                gridc = (a1, a0) if risc else (a0, a1)
                xr, _ = inj.get((x, 0), (0, 0)) if y == 0 else (0, 0)
                # single_root: only column 0's root sends, so ONE 52 KB whole-grid multicast is
                # timed with no inter-sender racing — the racing is what makes the all-roots figure
                # spread 27-60 us and swamp any posted-vs-nonposted effect.
                root = 1 if (y == 0 and (x == 0 or not single_root)) else 0
                args = [
                    x,
                    y,
                    m_eff,
                    hn_sizes[x],
                    kr_sizes[y],
                    xr,
                    root,
                    rowc[0][0],
                    rowc[0][1],
                    rowc[1][0],
                    rowc[1][1],
                    gridc[0][0],
                    gridc[0][1],
                    gridc[1][0],
                    gridc[1][1],
                ]
                for oy in range(kgroups):  # my column's cores, root first
                    ox, oyy = phys(x, oy)
                    args += [ox, oyy]
                rt[x][y] = args
        rts.append(rt)

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "dl_collective.cpp"),
            core_ranges=cores,
            compile_time_args=ct_for(0, 0, 1),
            runtime_args=rts[0],
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "dl_collective.cpp"),
            core_ranges=cores,
            compile_time_args=ct_for(1, 2, 3),
            runtime_args=rts[1],
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def volumes(p, m_eff, phases, assign=None):
    assign = assign or OP_ASSIGN
    hgroups, kgroups = p["hgroups"], p["kgroups"]
    hn_sizes, _ = p["hn"]
    kr_sizes, _ = p["kr"]
    n = lambda ph: sum(assign[ph])  # how many RISCs actually send it
    total = 0
    if "x" in phases:
        total += n("x") * sum(m_eff * kr_sizes[y] * BFP8_TILE * (hgroups - 1) for y in range(kgroups))
    if "reduce" in phases:
        for x in range(hgroups):
            blk = m_eff * hn_sizes[x] * BFP8_TILE
            sl = blk // kgroups
            total += n("reduce") * (kgroups * (kgroups * 2 * sl) + kgroups * 2 * sl)
    if "h" in phases:
        total += n("h") * sum(m_eff * hn_sizes[x] * BFP8_TILE * (hgroups * kgroups - 1) for x in range(hgroups))
    return total


def measure(
    device,
    *,
    count=256,
    emb=7168,
    phases=("x", "reduce", "h"),
    assign=None,
    posted=0,
    single_root=False,
    hgroups=11,
    kgroups=8,
    reps=5,
):
    hgroups, kgroups = base.clamp_grid(device, hgroups, kgroups)
    m_eff = min(-(-count // TILE), 8)
    p = base.plan(device, emb, hgroups, kgroups, -(-count // TILE))
    program = build(device, p, phases=phases, m_eff=m_eff, assign=assign, posted=posted, single_root=single_root)
    dummy_in = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, TILE]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    dummy_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, TILE]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    io = [dummy_in, dummy_out]

    ttnn.synchronize_device(device)
    base.read_kernel_ns(device)
    ttnn.generic_op(io, program)
    base.read_kernel_ns(device)
    samples = []
    for _ in range(reps):
        ttnn.generic_op(io, program)
        ns = base.read_kernel_ns(device)
        if ns:
            samples.append(ns)
    samples.sort()
    med = samples[len(samples) // 2]
    nbytes = volumes(p, m_eff, phases, assign)
    if single_root and "h" in phases:  # only one root sent
        nbytes = m_eff * max(p["hn"][0]) * BFP8_TILE * (hgroups * kgroups - 1)
    return {
        "phases": "+".join(phases) if phases else "none",
        "posted": posted,
        "bytes": nbytes,
        "ns_median": med,
        "ns_min": samples[0],
        "ns_max": samples[-1],
        "gbps": nbytes / (med * 1e-9) / 1e9 if nbytes else 0.0,
        "cores": hgroups * kgroups,
    }
