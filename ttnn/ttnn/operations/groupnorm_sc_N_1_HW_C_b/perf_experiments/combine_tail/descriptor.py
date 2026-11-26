# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""perf_experiments/combine_tail: the op's create_program_descriptor with the combine-tail options (flags):
  ar   all-reduce: every core is a compute/writer "root"; the writer gets its unit's mcast rectangles; no ttnn.Mcast
  mr   cb_group_stats is 3 pages ([rstd, mr, e])
  au   ar with a unicast all-gather (peer list) instead of the rectangle multicasts
  fraw raw finalize: cb_group_shift is a matmul operand (no UnpackToDestFp32); no ones tiles are built
Every flag becomes a CT_<FLAG> define on every kernel. No flags = the op's descriptor (kernel_dir aside).
"""
from pathlib import Path

import ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor import *  # noqa: F401,F403
from ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor import (
    _cb,
    _ceil_div,
    _f32_bits,
    _tuning,
    _used_core_range_set,
)


def _unit_rects(plan, wk):
    """The logical cores of wk's unit as row-major rectangles (sharded: one grid-column segment)."""
    if plan.sharded:
        x = wk.core.x
        y0 = wk.root.y
        return [(x, y0, x, y0 + plan.cpi - 1)]
    gx = plan.grid_x
    first = wk.root.y * gx + wk.root.x
    last = first + plan.cpi - 1
    rects = []
    c = first
    while c <= last:
        y, x = divmod(c, gx)
        if x == 0 and last - c + 1 >= gx:
            rows = (last - c + 1) // gx
            rects.append((0, y, gx - 1, y + rows - 1))
            c += rows * gx
        else:
            x_end = min(gx - 1, x + (last - c))
            rects.append((x, y, x_end, y))
            c += x_end - x + 1
    return rects


def _ar_rect_args(plan, device, wk, sw):
    """[n_rects, (sx, sy, ex, ey, cores, has_self) * n]: virtual coords, NOC1 order (end corner first) unless the
    writer is on NOC0 (sw: the swapped DRAM odd-column cores)."""
    out = []
    rects = _unit_rects(plan, wk)
    assert len(rects) <= 3, rects
    writer_noc1 = not sw
    for x0, y0, x1, y1 in rects:
        lo = device.worker_core_from_logical_core(ttnn.CoreCoord(x0, y0))
        hi = device.worker_core_from_logical_core(ttnn.CoreCoord(x1, y1))
        if writer_noc1:
            sx, sy, ex, ey = hi.x, hi.y, lo.x, lo.y
        else:
            sx, sy, ex, ey = lo.x, lo.y, hi.x, hi.y
        cores = (x1 - x0 + 1) * (y1 - y0 + 1)
        has_self = int(x0 <= wk.core.x <= x1 and y0 <= wk.core.y <= y1)
        out += [sx, sy, ex, ey, cores, has_self]
    return [len(rects)] + out


def _au_peer_args(plan, device, wk):
    """[n_peers, (vx << 16 | vy) * n]: every core of wk's unit except wk itself (virtual coords)."""
    peers = []
    for x0, y0, x1, y1 in _unit_rects(plan, wk):
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if (x, y) == (wk.core.x, wk.core.y):
                    continue
                v = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
                peers.append((v.x << 16) | v.y)
    assert len(peers) == plan.cpi - 1, (len(peers), plan.cpi)
    return [len(peers)] + peers


def create_program_descriptor(
    plan, input_tensor, output_tensor, *, gamma=None, beta=None, eps=1e-5, flags=(), kernel_dir=None
):
    GRAD = "grad" in flags  # the graduation package: graduation/ kernels (no defines) = mr + fraw, dead CBs dropped
    if GRAD:
        flags = ("mr", "fraw")
        kernel_dir = Path(kernel_dir).parent / "graduation"
    AU = "au" in flags
    AR, MR, FRAW = "ar" in flags or AU, "mr" in flags, "fraw" in flags
    if AU and "ar" not in flags:
        flags = tuple(flags) + ("ar",)
    KERNEL_DIR = Path(kernel_dir)
    device = input_tensor.device()
    W, T, R, h = plan.W, plan.T, plan.R, plan.h
    is_rm = plan.is_rm
    hg, hb = int(plan.has_gamma), int(plan.has_beta)
    bf16 = ttnn.bfloat16
    f32 = ttnn.float32

    sharded = plan.sharded
    cores = plan.shard_grid if sharded else _used_core_range_set(len(plan.work), plan.grid_x)
    map_period = plan.map_period

    # In place the output tensor is the input tensor, so the output CBs below borrow the input shard's bytes.
    assert output_tensor is input_tensor or not plan.in_place

    def shard_cb(tensor, formats):
        # Zero-copy: the CB is the core's own shard (no NoC traffic); the views are tile-sized pages over it.
        d = ttnn.cb_descriptor_from_sharded_tensor(formats[0][0], tensor)
        d.format_descriptors = [
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page) for idx, fmt, page in formats
        ]
        return d

    # ---------------- Circular buffers ----------------
    cbs = []
    if is_rm:
        if sharded:
            cbs.append(shard_cb(input_tensor, [(CB_INPUT_RM, bf16, BF16_TILE_BYTES)]))
            cbs.append(shard_cb(output_tensor, [(CB_OUTPUT_RM, bf16, BF16_TILE_BYTES)]))
        else:
            cbs.append(_cb(D_RM * T * BF16_TILE_BYTES, cores, [(CB_INPUT_RM, bf16, BF16_TILE_BYTES)]))
    # Alias group A: the input ring, K_alloc resident chunks + the streaming depth, as two FIFO views pushed in
    # lockstep. Pass 1 pops every chunk from CB_INPUT_TILES; CB_INPUT_KEEP drops the streamed head in pass 1 and
    # holds the resident tail (then the re-read head) for pass 2. Sharded TILE: the ring is the shard (K = B).
    ring_views = [(CB_INPUT_TILES, bf16, BF16_TILE_BYTES), (CB_INPUT_KEEP, bf16, BF16_TILE_BYTES)]
    if sharded and not is_rm:
        cbs.append(shard_cb(input_tensor, ring_views))
        cbs.append(shard_cb(output_tensor, [(CB_OUTPUT_TILES, bf16, BF16_TILE_BYTES)]))
    else:
        cbs.append(
            _cb(
                input_ring_pages(T=T, K_alloc=plan.K_alloc, stream=plan.stream, is_rm=is_rm) * BF16_TILE_BYTES,
                cores,
                ring_views,
            )
        )
        if not sharded:
            # Sharded RM pack-untilizes the normalized tiles straight into the output shard: no tile ring.
            cbs.append(_cb(output_ring_bytes(T), cores, [(CB_OUTPUT_TILES, bf16, BF16_TILE_BYTES)]))
    # Alias group E: the S1' spill accumulator (UnpackToDestFp32, multi-chunk pass 1) / the FPU-readable chan_scale.
    # The last chunk reloads accumulator tile i and packs the collapsed chan tile i over it (in-place lockstep; every
    # view returns to its base each block), so the accumulator costs no L1 of its own.
    cbs.append(
        _cb(
            W * F32_TILE_BYTES,
            cores,
            [(CB_SUM_ACC, f32, F32_TILE_BYTES), (CB_CHAN_SCALE, f32, F32_TILE_BYTES)],
        )
    )
    # Alias group C: the T2 accumulator (pass 1) / chan_work (b', expand -> pass 2), disjoint lifetimes.
    cbs.append(
        _cb(
            W * F32_TILE_BYTES,
            cores,
            [(CB_SUMSQ_ACC, f32, F32_TILE_BYTES), (CB_CHAN_WORK, f32, F32_TILE_BYTES)],
        )
    )
    cbs.append(_cb(W * BF16_TILE_BYTES, cores, [(CB_GROUP_MAP, bf16, BF16_TILE_BYTES)]))
    cbs.append(_cb(W * BF16_TILE_BYTES, cores, [(CB_PILOT_ROWS, bf16, BF16_TILE_BYTES)]))
    cbs.append(_cb(BF16_TILE_BYTES, cores, [(CB_NEG_ONE, bf16, BF16_TILE_BYTES)]))
    cbs.append(_cb(F32_TILE_BYTES, cores, [(CB_GROUP_SHIFT, f32, F32_TILE_BYTES)]))
    if hg:
        # The expand matmul's in1: the gamma map, + the beta map when beta (reader-built; beta-only is refused).
        cbs.append(_cb((hg + hb) * W * BF16_TILE_BYTES, cores, [(CB_GAMMA_ROWS, bf16, BF16_TILE_BYTES)]))
    # Alias group D: local partials (collapse -> gather) / [mean*rstd, e] (expand). The writer pops the partials
    # after its gather write, before the stats multicast the expand waits for can arrive; the next collapse packs
    # only after this block's pass 2.
    cbs.append(
        _cb(
            2 * F32_TILE_BYTES,
            cores,
            [(CB_LOCAL_PARTIALS, f32, F32_TILE_BYTES)] + ([] if GRAD else [(CB_MR_E, f32, F32_TILE_BYTES)]),
        )
    )
    # Remote-written landing zones (allocated on every used core so the base address is identical).
    cbs.append(_cb(2 * R * F32_TILE_BYTES, cores, [(CB_GATHERED_PARTIALS, f32, F32_TILE_BYTES)]))
    if not GRAD:
        cbs.append(_cb(R * BF16_TILE_BYTES, cores, [(CB_ONES_TILES, bf16, BF16_TILE_BYTES)]))
    cbs.append(_cb(2 * F32_TILE_BYTES, cores, [(CB_STATS_OUT, f32, F32_TILE_BYTES)]))
    cbs.append(_cb((3 if MR else 2) * F32_TILE_BYTES, cores, [(CB_GROUP_STATS, f32, F32_TILE_BYTES)]))
    stage = staging_bytes(W=W, map_period=map_period, is_rm=is_rm, n_affine=hg + hb)
    cbs.append(_cb(stage, cores, [(CB_AFFINE_STAGING, bf16, stage)]))

    # ---------------- Kernel args ----------------
    in_page_bytes = input_tensor.buffer_page_size()
    out_page_bytes = output_tensor.buffer_page_size()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0
    beta_addr = beta.buffer_address() if beta is not None else 0

    reader_ct = [W, h, int(is_rm), hg, hb, R, plan.Cg, plan.Wt, plan.Ht, plan.gx, in_page_bytes, map_period]
    reader_ct.append(int(sharded))
    # Per-chunk zones (CT): every core's chunk steps fit the marker budget.
    chunk_zones = int(
        bool(_tuning().get("chunk_zones"))
        and plan.B_max + (plan.B_max - min(plan.K_alloc, plan.B_max)) <= CHUNK_ZONE_MAX_STEPS
    )
    reader_ct.append(chunk_zones)
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(
        (ttnn.TensorAccessorArgs(gamma) if gamma is not None else ttnn.TensorAccessorArgs()).get_compile_time_args()
    )
    reader_ct.extend(
        (ttnn.TensorAccessorArgs(beta) if beta is not None else ttnn.TensorAccessorArgs()).get_compile_time_args()
    )

    # Compute is built once per role too (IS_ROOT is compile-time, CT 7): a runtime role flag lets O3 unswitch the
    # block loop on it, emitting the whole per-block body twice; the root binary must fit the kernel-config ring
    # (also under the watcher / assert build).
    def compute_ct(is_root):
        return [
            W,
            h,
            int(is_rm),
            hg,
            hb,
            R,
            int(sharded),
            int(is_root),
            chunk_zones,
        ]

    # The writer is built once per role (IS_ROOT is compile-time): each binary carries only its own mcast pipe
    # (sender on unit roots, receiver elsewhere) and only the root's gather-wait / zero-fill code.
    def writer_ct(is_root):
        ct = [W, h, R, plan.Wt, plan.Ht, plan.gx, out_page_bytes, int(is_root), int(is_rm), int(sharded), chunk_zones]
        ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
        return ct

    inv_n_bits = _f32_bits(1.0 / (plan.HW * plan.Cg))
    eps_bits = _f32_bits(eps)

    # NoC pairing per core. Every core reads on one NoC and writes on the other; DRAM interleaved, the odd grid
    # columns take the swapped pairing (read NOC1 / write NOC0), so both NoCs carry half of every phase's DRAM
    # stream (pass-1 reads, pass-2 reads + writes) instead of all reads on NOC0 and all writes on NOC1. Unit roots
    # keep (NOC0, NOC1): the mcast sender's NoC must be the McastConfig NoC. Sharded inputs stream nothing.
    def swapped(wk):
        return int(not sharded and not wk.is_root and wk.core.x % 2 == 1)

    reader_rt = {0: ttnn.RuntimeArgs(), 1: ttnn.RuntimeArgs()}
    reader_cores = {0: [], 1: []}
    compute_rt = {1: ttnn.RuntimeArgs(), 0: ttnn.RuntimeArgs()}
    writer_rt = {(r, sw): ttnn.RuntimeArgs() for r in (1, 0) for sw in (0, 1)}
    writer_cores = {1: [], 0: []}
    writer_cores_sw = {(r, sw): [] for r in (1, 0) for sw in (0, 1)}
    for wk in plan.work:
        x, y = wk.core.x, wk.core.y
        sw = swapped(wk)
        B_core = _ceil_div(wk.rows_core, h)
        K_core = min(plan.K_alloc, B_core)
        root_virt = device.worker_core_from_logical_core(wk.root)
        reader_cores[sw].append(wk.core)
        reader_rt[sw][x][y] = [
            input_tensor.buffer_address(),
            gamma_addr,
            beta_addr,
            wk.unit_start,
            wk.num_blocks,
            wk.row_start,
            wk.rows_core,
            K_core,
            1 if AR else wk.is_root,
            root_virt.x,
            root_virt.y,
        ]
        role = 1 if AR else wk.is_root
        compute_rt[role][x][y] = [wk.num_blocks, wk.rows_core, K_core, inv_n_bits, eps_bits]
        writer_cores[role].append(wk.core)
        writer_cores_sw[(role, sw)].append(wk.core)
        writer_rt[(role, sw)][x][y] = [
            output_tensor.buffer_address(),
            wk.unit_start,
            wk.num_blocks,
            wk.row_start,
            wk.rows_core,
            K_core,
            wk.slot,
            root_virt.x,
            root_virt.y,
            plan.cpi,
        ] + (_au_peer_args(plan, device, wk) if AU else _ar_rect_args(plan, device, wk, sw) if AR else [])

    reader_kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_b_reader.cpp"),
            core_ranges=(
                cores if not reader_cores[1] else ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in reader_cores[sw]])
            ),
            compile_time_args=reader_ct,
            runtime_args=reader_rt[sw],
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1 if sw else ttnn.NOC.NOC_0
            ),
        )
        for sw in (0, 1)
        if reader_cores[sw]
    ]
    writer_kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_b_writer.cpp"),
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in writer_cores_sw[(role, sw)]]),
            compile_time_args=writer_ct(role),
            runtime_args=writer_rt[(role, sw)],
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0 if sw else ttnn.NOC.NOC_1
            ),
        )
        for role in (1, 0)
        for sw in (0, 1)
        if writer_cores_sw[(role, sw)]
    ]
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi3,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=False,
    )
    unpack_modes = [ttnn.UnpackToDestMode.Default] * NUM_CB_SLOTS
    for idx in (CB_SUM_ACC, CB_SUMSQ_ACC) + (() if FRAW else (CB_GROUP_SHIFT,)):
        unpack_modes[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_config.unpack_to_dest_mode = unpack_modes
    compute_kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_b_compute.cpp"),
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in writer_cores[role]]),
            compile_time_args=compute_ct(role),
            runtime_args=compute_rt[role],
            config=compute_config,
        )
        for role in (1, 0)
        if writer_cores[role]
    ]

    gather_sem = ttnn.SemaphoreDescriptor(id=GATHER_SEMAPHORE_ID, core_ranges=cores, initial_value=0)
    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=[gather_sem], cbs=cbs)

    # Stats multicast: one channel over all used cores, groups of cpi consecutive cores, sender = slot 0 (the
    # unit root). DRAM: units are row-major core runs; sharded: a unit is a grid-column segment (ColumnMajor).
    # cpi == 1 degenerates to the sender's local copy.
    defines = [] if GRAD else [(f"CT_{f.upper()}", "1") for f in flags]
    for k in [*reader_kernels, *writer_kernels, *compute_kernels]:
        k.defines = list(k.defines) + defines
    if AR:
        descriptor.kernels = [*reader_kernels, *writer_kernels, *compute_kernels]
        return descriptor
    mcast = ttnn.Mcast(
        device,
        ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=True),
        cores,
        plan.cpi,
        ttnn.McastFixedSenderConfig(sender_index=0),
        ttnn.McastCoreOrder.ColumnMajor if sharded else ttnn.McastCoreOrder.RowMajor,
    )
    mcast.attach(descriptor, "mcast", writer_kernels, MCAST_FIRST_SEMAPHORE_ID)
    descriptor.kernels = [*reader_kernels, *writer_kernels, *compute_kernels]
    return descriptor
