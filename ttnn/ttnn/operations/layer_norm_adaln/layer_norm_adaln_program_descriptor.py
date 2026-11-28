# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for layer_norm_adaln.

Work distribution (op_design.md "Work Distribution"): `num_row_groups * num_w_slices` cores taken
row-major from (0,0); core linear id L -> group g = L // S, slice s = L % S. Each group owns a
contiguous, balanced range of tile rows; slice s owns slice_widths(Wt, S)[s] consecutive W tiles (ragged when S does
not divide Wt: the first Wt % S slices are one tile wider; kernels compile per width class). Per-slice
row statistics are all-gathered within a group (1xS run inside one grid row) by the writer's one-shot
streamed unicast schedule (compact payload prefix), then every member combines them redundantly (Chan's parallel combine).
Every core of grid column x owns slice x % S (S divides grid_x), so the affine w / b row-0 tiles are fetched once per
column by its top core's NoC1 RISC (the writer, or the reader on the NoC-swapped east columns) and multicast down the
column (raw semaphore flag; see layer_norm_adaln_affine.hpp).

Every CB size / trip count derives from the `Blocking` knobs chosen by `select_blocking`.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import ttnn

from .layer_norm_adaln import (
    exchange_oneshot,
    gather_depth,
    l1_cb_budget_bytes,
    INPUT_BLOCK_DEPTH,
    OUTPUT_ROW_DEPTH,
    select_blocking,
    slice_widths,
    stagger_phases,
    stats_payload_tiles,
    wbp_tiles,
)

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB slots (semantic names; the numeric slot is just the buffer index).
CB_INPUT_TILES = 0
CB_REDUCE_SCALER = 1
CB_PIVOT = 3
CB_LOCAL_STATS = 5
CB_GATHERED_STATS = 6
CB_ROW_STATS = 7
CB_WEIGHT_ROWS = 10
CB_BIAS_ROWS = 11
CB_OUTPUT_TILES = 14
CB_WBP = 12  # affine: w / b P tiles for the SFPU affine (3 columns per tile)
CB_NSTATS = 13  # 16-bit DEST: per-row rstd / c'' strip tiles for the SFPU normalize

STATS_CHUNK = 4  # reader push / stats streaming granularity within a tile row (Perf 2 stats_chain)
AFFINE_SEM_ID = 0  # data-ready flag of the affine column multicast; the stats channel's semaphores follow it
AFFINE_RECEIVER, AFFINE_INJECTOR, AFFINE_ALONE = 0, 1, 2  # writer roles (layer_norm_adaln_writer.cpp)
# Per-core NoC swap (Perf 4 noc_swap): the easternmost grid columns run the reader on NoC1 and the writer on NoC0
# (everyone else: reader NoC0, writer NoC1; each RISC still owns one NoC, so dedicated-NoC mode stays legal). On BH
# these columns sit just west of the NoC x-wrap into the west DRAM column: their NoC1 read responses and NoC0 writes
# take the short way round, and their reads leave NoC0 (relieving the top rows' NoC0 read tail) while their writes
# leave NoC1 (removing the far-right write tail). Two columns is the hop-model boundary on the measured boards
# (physical x >= 14 of 0..16); a third column (physical 13, the hop tie) regressed 2304x3072 noaff +10% on one board.
NOC_SWAP_EAST_COLS = 2
_DRAM_EP_ROWS = {}


def _dram_write_endpoint_rows(device, noc):
    """Logical grid rows of the DRAM banks' `noc` endpoints (each bank's optimal worker sits on its endpoint row)."""
    key = (device.id(), str(noc))
    if key not in _DRAM_EP_ROWS:
        cores = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, noc)
        _DRAM_EP_ROWS[key] = frozenset(c.y for c in cores)
    return _DRAM_EP_ROWS[key]


def _perf_zone_defines():
    """Permanent per-stage zones (kernel_lib/perf_instrumentation.hpp) compile in only with the device profiler AND
    KERNEL_PERF_ZONES. LN_ADALN_PERF_ZONES=1 turns them on for a breakdown run; LN_ADALN_PERF_ZONES_TRISC=0|1|2 keeps
    the compute zones on one TRISC. Off by default so headline measurements are unperturbed."""
    if os.environ.get("LN_ADALN_PERF_ZONES", "0") in ("", "0"):
        return []
    out = [("KERNEL_PERF_ZONES", "1")]
    if os.environ.get("LN_ADALN_PERF_ZONES_TRISC"):
        out.append(("KERNEL_PERF_ZONES_TRISC", os.environ["LN_ADALN_PERF_ZONES_TRISC"]))
    return out


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _cb(index, num_pages, page_size, data_format, core_grid):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def _core_ranges_row_major(num_cores, grid_x):
    """CoreRangeSet of the first `num_cores` cores in row-major order from (0,0)."""
    full_rows, rem = divmod(num_cores, grid_x)
    ranges = []
    if full_rows > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if rem > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _row_spans(coords):
    """CoreRangeSet of a set of logical (x, y) cores: one range per contiguous x-span of each grid row."""
    ranges = []
    for y in sorted({c[1] for c in coords}):
        xs = sorted(c[0] for c in coords if c[1] == y)
        start = xs[0]
        for i, x in enumerate(xs):
            if i + 1 == len(xs) or xs[i + 1] != x + 1:
                ranges.append(ttnn.CoreRange(ttnn.CoreCoord(start, y), ttnn.CoreCoord(x, y)))
                if i + 1 < len(xs):
                    start = xs[i + 1]
    return ttnn.CoreRangeSet(ranges)


def create_program_descriptor(input_tensor, weight, bias, output_tensor, epsilon, compute_config):
    device = input_tensor.device()
    has_affine = weight is not None
    fp32_dest = bool(compute_config.fp32_dest_acc_en)

    shape = list(input_tensor.shape)
    H, W = shape[-2], shape[-1]
    nc = 1
    for d in shape[:-2]:
        nc *= d
    tensor_h_tiles = nc * ((H + 31) // 32)
    tensor_w_tiles = (W + 31) // 32

    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = grid.x, grid.y

    acc_dtype = ttnn.float32 if fp32_dest else ttnn.bfloat16
    acc_tile_bytes = ttnn.tile_size(acc_dtype)
    t16 = ttnn.tile_size(ttnn.bfloat16)
    ep_rows = _dram_write_endpoint_rows(device, ttnn.NOC.NOC_1)

    blk = select_blocking(
        tensor_h_tiles,
        tensor_w_tiles,
        grid_x,
        grid_y,
        has_affine=has_affine,
        acc_tile_bytes=acc_tile_bytes,
        fp32_dest_acc_en=fp32_dest,
        l1_budget_bytes=l1_cb_budget_bytes(device),
        stagger_ep_rows=ep_rows,
    )
    spr = stats_payload_tiles(fp32_dest)
    S = blk.num_w_slices
    G = blk.num_row_groups
    bh = blk.block_height
    bw = blk.block_width  # widest slice (ragged slices: CBs are sized for it on every core)
    # wide_groups: ragged slices when S does not divide Wt. Kernels compile per slice-width class (block_width is a
    # compile-time knob everywhere); every core of grid column x owns slice x % S (S divides grid_x), so a column is one
    # class and the affine column multicast is unchanged. The combine weights each slot by its slice's element count.
    widths = slice_widths(tensor_w_tiles, S)
    w_starts = [sum(widths[:j]) for j in range(S)]
    ragged = len(set(widths)) > 1
    width_classes = sorted(set(widths))
    num_cores = G * S
    core_grid = _core_ranges_row_major(num_cores, grid_x)
    oneshot = exchange_oneshot(S, bh, fp32_dest, acc_tile_bytes)
    # Every S > 1 blocking is one-shot: the compact payload prefix (bh * 512 B fp32) times S - 1 stays under
    # ONESHOT_MAX_BYTES whenever S * bh <= GATHER_TILE_CAP (select_blocking feasibility). The rotating ttnn.Mcast
    # schedule was deleted (Perf 1, 6th tournament: all-one-shot measured -6.4% on 8192x3072 wb fp32, flat elsewhere).
    assert S == 1 or oneshot, (S, bh)
    depth = gather_depth(S, bh, blk.num_blocks_per_core, fp32_dest, acc_tile_bytes)

    # ---------------- Circular buffers (sizes derive from bh / bw / S / depths) ----------------
    # Ragged slices: the input / output CBs are streamed in per-row pushes of the core's own width, so their capacity
    # must be a multiple of it (a ring sized for the widest slice would split a narrow core's row across the wrap).
    # They get one descriptor per width class, allocated LAST: every CB before them is uniform, so cb_gathered_stats
    # (the exchange's remote landing address) and the affine landing CBs sit at the same L1 address on every core.
    cbs = [
        _cb(CB_REDUCE_SCALER, 1, t16, ttnn.bfloat16, core_grid),
        _cb(CB_LOCAL_STATS, spr * bh, acc_tile_bytes, acc_dtype, core_grid),
        _cb(CB_GATHERED_STATS, spr * S * bh * depth, acc_tile_bytes, acc_dtype, core_grid),
        _cb(CB_ROW_STATS, 2 * bh, acc_tile_bytes, acc_dtype, core_grid),
    ]

    def stream_cbs(w, grid):
        return [
            _cb(CB_INPUT_TILES, INPUT_BLOCK_DEPTH * bh * w, t16, ttnn.bfloat16, grid),
            _cb(CB_OUTPUT_TILES, OUTPUT_ROW_DEPTH * w, t16, ttnn.bfloat16, grid),
        ]

    if not ragged:
        cbs = stream_cbs(bw, core_grid)[:1] + cbs + stream_cbs(bw, core_grid)[1:]  # HEAD order
    if has_affine:
        # Resident w / b row-0 tiles (the column multicast lands them); normalize Row-broadcasts row 0 straight from
        # srcB, so no full-tile copy exists.
        cbs += [
            _cb(CB_WEIGHT_ROWS, bw, t16, ttnn.bfloat16, core_grid),
            _cb(CB_BIAS_ROWS, bw, t16, ttnn.bfloat16, core_grid),
            _cb(CB_WBP, wbp_tiles(bw, fp32_dest), t16, ttnn.bfloat16, core_grid),
        ]
    if not fp32_dest:
        cbs.append(_cb(CB_NSTATS, 2 if has_affine else 1, t16, ttnn.bfloat16, core_grid))

    # ---------------- Per-core work assignment ----------------
    # Per-core DRAM bank stagger: every core visits its row's block_width tiles starting at column rot = L % num_banks
    # and wraps (reader, normalize and writer all use the same order), so the cores' first pages spread over the banks
    # instead of all hitting bank 0 and walking the banks in lockstep (Perf 2 dram_bw).
    num_dram_banks = ttnn.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
    base_rows, extra_rows = divmod(tensor_h_tiles, G)
    per_core = []  # (core, group_rows, row_start, w_start, rot, width)
    width_of = {}
    for L in range(num_cores):
        g, s = divmod(L, S)
        group_rows = base_rows + (1 if g < extra_rows else 0)
        row_start = g * base_rows + min(g, extra_rows)
        core = ttnn.CoreCoord(L % grid_x, L // grid_x)
        w = widths[s]
        width_of[(core.x, core.y)] = w
        per_core.append((core, group_rows, row_start, w_starts[s], (L % num_dram_banks) % w, w))

    # Affine row-0 fetch granularity: one DRAM-aligned chunk per face (faces 0 and 1 hold row 0).
    affine_fetch_bytes = max(32, ttnn.get_dram_alignment())

    # ---------------- Reader ----------------
    # stats_chain (Perf 2): the reader pushes a row in chunks of STATS_CHUNK tiles (one NoC read trid each, <= 15
    # chunks) so compute streams the row's stats behind the load; the pivot is the bf16 mean of the first chunk.
    def chunk_of(w):
        return min(w, max(STATS_CHUNK, -(-w // 15)))

    cbs.append(_cb(CB_PIVOT, bh, t16, ttnn.bfloat16, core_grid))
    reader_acc_ct = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    reader_acc_ct += ttnn.TensorAccessorArgs(weight if has_affine else input_tensor).get_compile_time_args()
    reader_acc_ct += ttnn.TensorAccessorArgs(bias if has_affine else input_tensor).get_compile_time_args()

    def reader_ct(w):
        return [bh, w, tensor_w_tiles, chunk_of(w)] + reader_acc_ct

    def class_sets(cs):
        # (width, cores of cs in that width class), non-empty only
        out = [(w, {xy for xy in cs if width_of[xy] == w}) for w in width_classes]
        return [t for t in out if t[1]]

    # NoC assignment (NOC_SWAP_EAST_COLS), every blocking. The Perf 2 carve-out (affine + 16-bit DEST with >= 4 tile rows
    # per core kept the default pair) was deleted: on the current kernels (east-column affine fetch on the reader) the
    # swap wins there too (perf_experiments/worksplit_refit: 23040x3072 wb 16-bit -6.3%, 8192x8192 -5.2%).
    swapped = {(c.x, c.y) for c, *_ in per_core if c.x >= grid_x - NOC_SWAP_EAST_COLS}
    kept = {(c.x, c.y) for c, *_ in per_core} - swapped
    noc_sets = [(kept, ttnn.NOC.NOC_0, ttnn.NOC.NOC_1), (swapped, ttnn.NOC.NOC_1, ttnn.NOC.NOC_0)]
    noc_sets = [t for t in noc_sets if t[0]]
    # one reader / writer kernel per (NoC assignment, slice-width class)
    kernel_sets = [(cs_all, rn, wn, w, cs) for cs_all, rn, wn in noc_sets for w, cs in class_sets(cs_all)]
    writer_noc_of = {xy: wn for cs, _, wn in noc_sets for xy in cs}

    # ---------------- Writer (stats all-gather + output store) ----------------
    writer_acc_ct = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    # w / b accessors are declared unconditionally in the kernel; when absent the output's args stand in as an
    # (unused) placeholder — this binding has no no-arg TensorAccessorArgs constructor.
    writer_acc_ct += ttnn.TensorAccessorArgs(weight if has_affine else output_tensor).get_compile_time_args()
    writer_acc_ct += ttnn.TensorAccessorArgs(bias if has_affine else output_tensor).get_compile_time_args()

    def writer_ct(w):
        return [bh, w, S, tensor_w_tiles, acc_tile_bytes, spr, int(has_affine), affine_fetch_bytes, AFFINE_SEM_ID] + (
            writer_acc_ct
        )

    # Affine column multicast: grid column x holds rows 0..h_x-1 (row-major fill, so the last grid row may be partial);
    # its top core injects to rows 1..h_x-1. Rectangle in virtual NoC coords, start/end ordered for the owning RISC's NoC.
    full_rows, rem = divmod(num_cores, grid_x)

    def affine_args(core, owner_noc):
        if not has_affine:
            return []
        h = full_rows + (1 if core.x < rem else 0)
        head = [weight.buffer_address(), bias.buffer_address()]
        if core.y > 0:
            return head + [AFFINE_RECEIVER]
        if h == 1:
            return head + [AFFINE_ALONE]
        a = device.worker_core_from_logical_core(ttnn.CoreCoord(core.x, 1))
        b = device.worker_core_from_logical_core(ttnn.CoreCoord(core.x, h - 1))
        hi, lo = (max(a.x, b.x), max(a.y, b.y)), (min(a.x, b.x), min(a.y, b.y))
        if owner_noc == ttnn.NOC.NOC_0:
            return head + [AFFINE_INJECTOR, lo[0], lo[1], hi[0], hi[1], h - 1]  # NoC 0: start = low corner
        return head + [AFFINE_INJECTOR, hi[0], hi[1], lo[0], lo[1], h - 1]  # NoC 1: start = high corner

    def exchange_args(core):
        # One-shot all-gather: own slot, the group's grid-row virtual y, every member's virtual x (slot order).
        if not oneshot:
            return []
        s = core.x % S
        vs = [device.worker_core_from_logical_core(ttnn.CoreCoord(core.x - s + j, core.y)) for j in range(S)]
        return [s, vs[0].y] + [v.x for v in vs]

    def affine_in_reader(cs):
        # The affine w / b column fetch + multicast runs on the core's NoC1 RISC: the writer on the default columns,
        # the reader on the NOC_SWAP_EAST_COLS columns (perf_experiments/affine_fetch_fix). There the NoC0 writer's
        # reads queued behind the chip-wide NoC0 x-read storm (landing at ~42 us on 2304x3072) and held the column's
        # c_wbp_prep and its group's exchange. S == 1 keeps the writer: no exchange waits on the affine, so a late
        # east fetch is pure slack, while fetching it early lets the east columns' NoC0 output writes overlap the west
        # columns' NoC0 read tail (8192x256 wb fp32 +8%).
        return int(has_affine and cs is swapped and S > 1)

    # read_stagger: phase-p group i, slice s waits on phase-(p-1) group i, slice s (phases never grow, so every
    # predecessor releases at most one successor). RT args [gated, has_successor, succ_x, succ_y], zeros when off.
    phases = None
    if blk.core_h_tiles == 1 and base_rows >= 1:
        phases = stagger_phases(G, S, grid_x, ep_rows, bw * t16, has_affine, fp32_dest)
    stagger = phases is not None
    stagger_sem_id = (1 if has_affine else 0) + (S if oneshot else 0)
    stagger_rt = {L: [0, 0, 0, 0] for L in range(num_cores)}
    if stagger:
        for prev, cur in zip(phases, phases[1:]):
            assert len(cur) <= len(prev), phases
            for g, gp in zip(cur, prev):
                for s_ in range(S):
                    L, Lp = g * S + s_, gp * S + s_
                    v = device.worker_core_from_logical_core(ttnn.CoreCoord(L % grid_x, L // grid_x))
                    stagger_rt[L][0] = 1
                    stagger_rt[Lp][1:4] = [1, v.x, v.y]

    reader_kernels = []
    for cs_all, rn, _, w, cs in kernel_sets:
        in_reader = affine_in_reader(cs_all)
        reader_rt = ttnn.RuntimeArgs()
        for core, group_rows, row_start, w_start, rot, _w in per_core:
            if (core.x, core.y) in cs:
                reader_rt[core.x][core.y] = (
                    [input_tensor.buffer_address(), row_start, group_rows, w_start, rot]
                    + stagger_rt[core.y * grid_x + core.x]
                    + (affine_args(core, rn) if in_reader else [])
                )
        reader_kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "layer_norm_adaln_reader.cpp"),
                defines=_perf_zone_defines(),
                core_ranges=_row_spans(cs),
                compile_time_args=reader_ct(w),
                named_compile_time_args=[
                    ("affine_in_reader", in_reader),
                    ("affine_fetch_bytes", affine_fetch_bytes),
                    ("affine_sem_id", AFFINE_SEM_ID),
                    ("stagger", int(stagger)),
                    ("stagger_sem_id", stagger_sem_id),
                ],
                runtime_args=reader_rt,
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=rn),
            )
        )

    # Semaphores: [affine flag][one-shot per-sender flags x S][read_stagger release flag].
    flag_sem_base = AFFINE_SEM_ID + 1 if has_affine else 0
    writer_kernels = []
    for cs_all, _, wn, w, cs in kernel_sets:
        writer_rt = ttnn.RuntimeArgs()
        for core, group_rows, row_start, w_start, rot, _w in per_core:
            if (core.x, core.y) in cs:
                writer_args = [output_tensor.buffer_address(), row_start, group_rows, w_start, rot]
                writer_rt[core.x][core.y] = writer_args + exchange_args(core) + affine_args(core, wn)
        writer_kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "layer_norm_adaln_writer.cpp"),
                named_compile_time_args=[
                    ("xchg_oneshot", int(oneshot)),
                    ("xchg_flag_sem_base", flag_sem_base),
                    ("affine_in_reader", affine_in_reader(cs_all)),
                ],
                defines=_perf_zone_defines(),
                core_ranges=_row_spans(cs),
                compile_time_args=writer_ct(w),
                runtime_args=writer_rt,
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=wn),
            )
        )

    # ---------------- Compute (one kernel per slice-width class) ----------------
    W_total = tensor_w_tiles * 32
    kcfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=compute_config.math_fidelity,
        math_approx_mode=bool(compute_config.math_approx_mode),
        fp32_dest_acc_en=fp32_dest,
    )
    eps_bits = _f32_bits(epsilon)
    compute_kernels = []
    for w, cs in class_sets({(c.x, c.y) for c, *_ in per_core}):
        n_slice = w * 32
        chunk = chunk_of(w)
        pivot_tiles = chunk
        compute_ct = [
            bh,
            w,
            S,
            int(has_affine),
            _f32_bits(n_slice),
            _f32_bits(1.0 / n_slice),
            # ragged: the fold accumulates n_j-weighted terms, so the finalize divides by the row length instead
            _f32_bits(1.0 / W_total if ragged else 1.0 / S),
            _f32_bits(1.0 / W_total),
            spr,
            int(not fp32_dest),  # payload [p, M2 | d] (16-bit DEST) instead of [r_hi, r_lo | M2_hi, M2_lo]
            chunk,
            pivot_tiles,
            _f32_bits(1.0 / (32 * pivot_tiles)),
            int(ragged),
        ]
        compute_rt = ttnn.RuntimeArgs()
        for core, group_rows, row_start, w_start, rot, _w in per_core:
            if (core.x, core.y) in cs:
                # ragged: slot q of cb_gathered_stats holds slice (my_slot + q) mod S (writer exchange_streamed)
                my = core.x % S
                n_pos = [_f32_bits(32 * widths[(my + q) % S]) for q in range(S)] if ragged else []
                compute_rt[core.x][core.y] = [group_rows, eps_bits, rot] + n_pos
        # No CB needs UnpackToDestFp32: the row accumulators never leave DEST (stats_publish_fuse) and the fp32 payload
        # is a tf32-exact hi/lo split copied through srcA (combine_lean).
        compute_kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "layer_norm_adaln_compute.cpp"),
                defines=_perf_zone_defines(),
                core_ranges=_row_spans(cs),
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=kcfg,
            )
        )

    semaphores = []
    if has_affine:
        semaphores.append(ttnn.SemaphoreDescriptor(id=AFFINE_SEM_ID, core_ranges=core_grid, initial_value=0))
    if oneshot:
        assert flag_sem_base == len(semaphores)
        semaphores += [
            ttnn.SemaphoreDescriptor(id=flag_sem_base + j, core_ranges=core_grid, initial_value=0) for j in range(S)
        ]
    if stagger:
        assert stagger_sem_id == len(semaphores)
        semaphores.append(ttnn.SemaphoreDescriptor(id=stagger_sem_id, core_ranges=core_grid, initial_value=0))
    if ragged:
        for w, cs in class_sets({(c.x, c.y) for c, *_ in per_core}):
            cbs += stream_cbs(w, _row_spans(cs))
    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=semaphores, cbs=cbs)
    descriptor.kernels = reader_kernels + writer_kernels + compute_kernels
    return descriptor
