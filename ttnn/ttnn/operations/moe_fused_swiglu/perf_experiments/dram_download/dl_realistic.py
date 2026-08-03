# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Realistic DM recreation — the op's whole data movement, no compute, barriers where needed.

STAGE 0 = one trailing barrier (the floor dl_stream/dl_collective measure).
STAGE 1 = the op's dependency order: per-GU-chunk weight barriers, x stage -> mcast ordering, the
          reduce's two phases, a barrier per h round, W_down prefetched per round, output write.

Comparing the two isolates the cost of the ORDERING from the cost of the bytes. Against the op's own
`ABLATE=skip_compute` (104.4 us at count 256 — all DM + collectives + the real rendezvous, no math),
what remains is the semaphore rendezvous, which is stage 2 and not built here.
"""

from pathlib import Path

import ttnn

from ttnn.operations.moe_fused_swiglu.perf_experiments.dram_download import dl_bench as base

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE, BFP4_TILE, BFP8_TILE, HIDDEN = 32, 576, 1088, 2048
GU_CHUNKS = 3  # the op's shipped value


def build(device, p, tensors, *, stage, m_eff):
    x_t, w = tensors
    wg_t, wu_t, wd_t = w
    hgroups, kgroups = p["hgroups"], p["kgroups"]
    hn_sizes, hn_starts = p["hn"]
    kr_sizes, kr_starts = p["kr"]
    ec_sizes, ec_starts = p["ec"]
    hid_t, emb_t = p["hid_t"], p["emb_t"]
    hn_max, kr_max, ec_max = max(hn_sizes), max(kr_sizes), max(ec_sizes)
    num_cores = hgroups * kgroups

    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(hgroups - 1, kgroups - 1))])
    out_t = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, m_eff * TILE, p["emb_t"] * TILE]),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    block = m_eff * hn_max * BFP8_TILE
    w_bytes = GU_CHUNKS * kr_max * max(hn_max // GU_CHUNKS, 1) * BFP4_TILE
    w_bytes = max(w_bytes, hid_t * ec_max * BFP4_TILE)
    land_bytes = max(block, 2 * kgroups * (block // kgroups), m_eff * TILE * kr_max * TILE * 2)
    out_bytes = m_eff * ec_max * BFP8_TILE

    def cb(index, total, page):
        # A CB's total size must be an exact multiple of its page size. The x landing region is sized in
        # BYTES (stick slices, 1792 B each) while the page here is a bfp8 tile (1088 B), so the raw byte
        # count is not a multiple of it — round UP rather than silently truncating.
        pages = max(1, -(-total // page))
        return ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat8_b, page_size=page)
            ],
        )

    # One private set per RISC: CB_W, CB_SRC, CB_LAND, CB_OUT.
    cbs = []
    for b in (0, 4):
        cbs += [
            cb(b + 0, w_bytes, BFP4_TILE),
            cb(b + 1, block, BFP8_TILE),
            cb(b + 2, land_bytes, BFP8_TILE),
            cb(b + 3, out_bytes, BFP8_TILE),
        ]

    def phys(cx, cy):
        c = device.worker_core_from_logical_core(ttnn.CoreCoord(cx, cy))
        return int(c.x), int(c.y)

    a0, a1 = phys(0, 0), phys(hgroups - 1, kgroups - 1)
    inj = base._injectors(p)

    kernels = []
    for risc in (0, 1):  # 0 = reader/NOC_0 (W_gate + x), 1 = writer/NOC_1 (W_up + W_down + out)
        base_cb = 0 if risc == 0 else 4
        wt = wg_t if risc == 0 else wu_t
        ct = [
            stage,
            1 if risc == 0 else 0,
            hgroups,
            kgroups,
            GU_CHUNKS,
            hid_t,
            emb_t,
            BFP4_TILE,
            BFP8_TILE,
            num_cores,
            x_t.buffer_page_size(),
            base_cb + 0,
            base_cb + 1,
            base_cb + 2,
            base_cb + 3,
        ]
        for t in (wt, wd_t, x_t, out_t):
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

        rt = ttnn.RuntimeArgs()
        for x in range(hgroups):
            for y in range(kgroups):
                flat = y * hgroups + x
                nr, fr = phys(0, y), phys(hgroups - 1, y)
                # Rect corner order is NoC-dependent: NOC_1 takes far->near.
                rowc = (fr, nr) if risc else (nr, fr)
                gridc = (a1, a0) if risc else (a0, a1)
                xr, xr0 = inj.get((x, 0), (0, 0)) if (y == 0 and risc == 0) else (0, 0)
                args = [
                    wt.buffer_address(),
                    wd_t.buffer_address(),
                    x_t.buffer_address(),
                    out_t.buffer_address(),
                    x,
                    y,
                    m_eff,
                    kr_starts[y],
                    kr_sizes[y],
                    hn_starts[x],
                    hn_sizes[x],
                    ec_starts[flat],
                    ec_sizes[flat],
                    xr,
                    xr0,
                    1 if y == 0 else 0,
                    rowc[0][0],
                    rowc[0][1],
                    rowc[1][0],
                    rowc[1][1],
                    gridc[0][0],
                    gridc[0][1],
                    gridc[1][0],
                    gridc[1][1],
                ]
                for oy in range(kgroups):
                    args += list(phys(x, oy))
                rt[x][y] = args

        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "dl_realistic.cpp"),
                core_ranges=cores,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor() if risc == 0 else ttnn.WriterConfigDescriptor(),
            )
        )
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs), out_t


def measure(device, *, count=256, emb=7168, stage=1, hgroups=11, kgroups=8, reps=7, tensors=None):
    hgroups, kgroups = base.clamp_grid(device, hgroups, kgroups)
    m_tiles = -(-count // TILE)
    m_eff = min(m_tiles, 8)
    p = base.plan(device, emb, hgroups, kgroups, m_tiles)
    if tensors is None:
        tensors = base.make_tensors(device, emb, 5120, "bf16_rm", "nd_shard", m_tiles)
    program, out_t = build(device, p, tensors, stage=stage, m_eff=m_eff)
    x_t, w = tensors
    io = [x_t, w[0], w[1], w[2], out_t]

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
    return {
        "stage": stage,
        "ns_median": samples[len(samples) // 2],
        "ns_min": samples[0],
        "ns_max": samples[-1],
        "cores": hgroups * kgroups,
    }
