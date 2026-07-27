# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""onorm ProgramDescriptor — CBs, kernels, work distribution, args.

Blocking Model (op_design.md §1).  Every block factor / buffer depth /
core-assignment below is a **named parameter with exactly one source of truth**.
Every CB page count, loop trip count and grid size is *derived* from those
parameters — never restated as a second literal and never taken from a whole-op
dimension (T, B, the full flat width).

Work unit = one **token-block** = ``TOKENS_PER_BLOCK`` tokens of one batch.
``B * ceil(T / TOKENS_PER_BLOCK)`` such blocks are spread over the whole compute
grid with ``row_wise=True`` from phase 1 — there is no single-core phase.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"


# ===========================================================================
# Blocking Model knobs — THE single source of truth (op_design.md §1.1 / §1.2)
# ===========================================================================
#
# Each of these is a tunable.  Nothing downstream re-states its value: CB page
# counts, kernel loop bounds and the grid all derive from them.

# --- block factors ---
TOKENS_PER_BLOCK = 32  # tokens per work unit (= one output tile-row: the re-tile floor)
NORM_CHUNK_TOKENS = 8  # tokens per normalize sub-pass (coarsest that fits L1, §6.2)
GATE_CHUNK_TILES = 64  # output tiles per gate-chain invocation (phase C block factor)

# Tiles per noc_async_read / noc_async_write group — ONE barrier per group, so
# this many transfers are in flight at once.  Used by BOTH dataflow halves (the
# reader's `o`/`gate` streams and the writer's output stream read the same CT arg),
# so raising it widens both sides together.
# MEASURED (Blackhole p150, 11x10 grid, tests/.../test_onorm_trials.py, 5-7
# interleaved trials/config, <=1% spread): raising 4 -> 8 together with
# DM_DEPTH 2 -> 4 is 1.164x at 2 cores, 1.163x at 4, 1.108x at 20, 1.061x at 110.
# The win shrinks as cores rise because the op approaches the DRAM roofline
# (~412 GB/s aggregate at B=8/T=640, ~80% of Blackhole peak), which is exactly the
# behaviour op_design.md §1.4 predicts.
DM_BLOCK_TILES = 8

# --- buffer depths (in block-factor units) ---
# DM_DEPTH deepens cb_gate_tiles / cb_out_tiles so the reader can prefetch (and the
# writer drain) a group while compute works on the previous one.  4 is measured
# above; DM_DEPTH=2 at DM_BLOCK_TILES=8 leaves 1.035x on the table.
DM_DEPTH = 4
# O_DEPTH deepens cb_o_tiles. Kept at 2: the reader is NOT the critical path at
# these settings (at the new defaults, B=1/T=640: NCRISC 83.5us vs 88.0us compute
# vs 93.1us kernel), and O_DEPTH=3 measured within noise while costing +128 KB of
# L1. This stays a live knob for a future shape where the reader IS the bound.
O_DEPTH = 2

# --- hardware tile geometry (not a knob) ---
TILE_H = 32
TILE_W = 32

# L1 headroom held back from the CB budget.  `get_max_worker_l1_unreserved_size()`
# reports the unreserved L1 span, but the statically-allocated CB region starts
# ABOVE a per-program base (kernel binaries, runtime args, profiler buffers), so
# the CB total that the runtime actually validates is larger than the sum of our
# pages.  Measured on Blackhole p150: a 741-page (1517568 B) CB set was reported by the
# runtime as growing to 1628928 B against a max L1 of 1572864 B — i.e. a 111360 B
# base, so the real CB ceiling is 1572864 - 111360 = 1461504 B, which is
# get_max_worker_l1_unreserved_size() - 70656.  Holding back 72 KB makes this
# file's budget assert — which names the knobs to lower, and in what order — fire
# BEFORE the runtime's own bare "beyond max L1 size" throw, without rejecting knob
# settings that genuinely do fit.
_L1_CB_BASE_RESERVE = 72 * 1024


# ===========================================================================
# Circular-buffer indices (semantic names; the number is just the slot)
# ===========================================================================

CB_O_TILES = 0  # reader  -> compute : head-major `o` tiles
CB_GATE_TILES = 1  # reader  -> compute : flat pre-sigmoid `gate` tiles
CB_WEIGHT = 2  # reader  -> compute : RMSNorm scale, held for the whole kernel
CB_SCALER = 8  # reader  -> compute : reduce scaler (1/V), held
CB_OUT_TILES = 16  # compute -> writer  : finished flat output tiles
CB_SUMSQ = 24  # compute -> compute : per-token partial sum of squares
CB_RSTD = 25  # compute -> compute : rsqrt(mean + eps), col-0 valid
CB_NORMED = 27  # compute -> compute : o * rstd
CB_ONORM = 28  # compute -> compute : o * rstd * weight (untilize input)
CB_RM_FLAT_ROWS = 29  # compute -> compute : ROW-MAJOR flat feature rows (re-tile working set)
CB_FLAT_TILES = 30  # compute -> compute : flat token-major tiles
CB_GATE_SIG = 31  # compute -> compute : sigmoid(gate), materialised so the FPU feeds off L1


def _div_up(a: int, b: int) -> int:
    """Ceiling division (ttnn.div_up is not exposed in every build)."""
    return (a + b - 1) // b


def _f32_bits(value: float) -> int:
    """fp32 bit pattern of `value`, as the kernels' scalar ops consume it."""
    return struct.unpack("I", struct.pack("f", float(value)))[0]


def _compute_config_descriptor(compute_kernel_config) -> ttnn.ComputeConfigDescriptor:
    """Translate a ttnn.DeviceComputeKernelConfig into a ComputeConfigDescriptor.

    ``ttnn.generic_op`` takes a ``ComputeConfigDescriptor``; the public parameter
    is a ``DeviceComputeKernelConfig`` (a ``WormholeComputeKernelConfig`` in
    practice).  This is the one place the two are bridged, field by field.
    ``packer_l1_acc`` / ``throttle_level`` have no descriptor counterpart and are
    not used by this kernel.
    """
    math_fidelity = compute_kernel_config.math_fidelity
    if math_fidelity == ttnn.MathFidelity.Invalid:
        # WormholeComputeKernelConfig's own default; normalise to the op's.
        math_fidelity = ttnn.MathFidelity.HiFi4

    return ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        math_approx_mode=bool(compute_kernel_config.math_approx_mode),
        fp32_dest_acc_en=bool(compute_kernel_config.fp32_dest_acc_en),
        dst_full_sync_en=bool(compute_kernel_config.dst_full_sync_en),
    )


def _grid_assignment(device, num_token_blocks):
    """Spread the token-blocks over the whole compute grid.

    ``row_wise=True`` is mandatory: the default column-major layout puts every
    core on the same shared NoC links (measured 2.91x slower on a DRAM<->DRAM
    stream).  Both ``split_work_to_cores`` and ``corerange_to_cores`` must use
    the same ordering, otherwise the running ``start_block`` lands on the wrong
    core.
    """
    grid_size = device.compute_with_storage_grid_size()
    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        blocks_per_core_g1,
        blocks_per_core_g2,
    ) = ttnn.split_work_to_cores(grid_size, num_token_blocks, row_wise=True)

    assignment = []
    start_block = 0
    for group, per_core in ((core_group_1, blocks_per_core_g1), (core_group_2, blocks_per_core_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start_block, per_core))
            start_block += per_core
    assert (
        start_block == num_token_blocks
    ), f"onorm: work split covered {start_block} of {num_token_blocks} token-blocks"
    return num_cores, all_cores, assignment


def create_program_descriptor(
    o: ttnn.Tensor,
    gate: ttnn.Tensor,
    weight: ttnn.Tensor,
    output: ttnn.Tensor,
    epsilon: float,
    compute_kernel_config,
) -> ttnn.ProgramDescriptor:
    device = o.device()

    # ================= 1. GEOMETRY, DERIVED FROM THE KNOBS =================
    batch, tokens, num_heads, head_dim = list(o.shape)
    flat_width = int(gate.shape[-1])

    v_tiles = _div_up(head_dim, TILE_W)  # column tiles per head-major image
    flat_tiles = _div_up(flat_width, TILE_W)  # column tiles per flat tile-row
    token_tile_rows = _div_up(tokens, TILE_H)  # `Tt`: gate/out tile-rows per batch

    tile_rows_per_block = TOKENS_PER_BLOCK // TILE_H
    blocks_per_batch = _div_up(tokens, TOKENS_PER_BLOCK)
    num_token_blocks = batch * blocks_per_batch

    # Per-block tile counts. The kernels derive these from the same knobs (as
    # `o_tiles_per_block` / `flat_tiles_per_block`); only the flat one is needed
    # host-side, for CB sizing.
    flat_tiles_per_block = tile_rows_per_block * flat_tiles
    norm_chunks_per_block = TOKENS_PER_BLOCK // NORM_CHUNK_TOKENS
    gate_chunks_per_block = flat_tiles_per_block // GATE_CHUNK_TILES

    # --- knob consistency (a violated knob relation is a silent wrong answer) ---
    assert TOKENS_PER_BLOCK % TILE_H == 0, "TOKENS_PER_BLOCK must be a multiple of the tile height"
    assert (
        TOKENS_PER_BLOCK % NORM_CHUNK_TOKENS == 0
    ), f"NORM_CHUNK_TOKENS={NORM_CHUNK_TOKENS} must divide TOKENS_PER_BLOCK={TOKENS_PER_BLOCK}"
    assert flat_tiles_per_block % GATE_CHUNK_TILES == 0, (
        f"GATE_CHUNK_TILES={GATE_CHUNK_TILES} must divide the block's " f"{flat_tiles_per_block} flat output tiles"
    )
    assert 1 <= DM_BLOCK_TILES <= 8, "DM_BLOCK_TILES is a 1..8 knob"
    assert DM_DEPTH >= 2 and O_DEPTH >= 2, "streaming depths must be >= 2 to overlap read with compute"
    # `o`'s token axis is un-padded (tiled dims are (HV, V)) while gate/out's IS
    # tile-padded (tiled dims are (T, FLAT)).  T % TOKENS_PER_BLOCK == 0 is what
    # makes the two views coincide; a partial last block would read past `o`.
    assert (
        tokens % TOKENS_PER_BLOCK == 0
    ), f"onorm: T={tokens} must be a multiple of TOKENS_PER_BLOCK={TOKENS_PER_BLOCK}"

    tile_bytes = o.buffer_page_size()  # every CB is one bf16 tile per page
    assert gate.buffer_page_size() == tile_bytes and output.buffer_page_size() == tile_bytes

    # ================= 2. WORK DISTRIBUTION =================
    _, all_cores, assignment = _grid_assignment(device, num_token_blocks)

    # ================= 3. CIRCULAR BUFFERS =================
    # Streaming input/output CBs get `DM_BLOCK_TILES * DM_DEPTH` (the
    # double-buffer knob); intermediates between two *sequential* compute
    # helpers get the full block they must hold (both helpers own all three
    # TRISCs, so they cannot pipeline).  Nothing here scales with B or T.
    cb_pages = {
        CB_O_TILES: v_tiles * NORM_CHUNK_TOKENS * O_DEPTH,
        CB_GATE_TILES: DM_BLOCK_TILES * DM_DEPTH,
        CB_WEIGHT: v_tiles,
        CB_SCALER: 1,
        CB_OUT_TILES: DM_BLOCK_TILES * DM_DEPTH,
        CB_SUMSQ: NORM_CHUNK_TOKENS,
        CB_RSTD: NORM_CHUNK_TOKENS,
        CB_NORMED: v_tiles * NORM_CHUNK_TOKENS,
        CB_ONORM: v_tiles * NORM_CHUNK_TOKENS,
        # EXACTLY one block's worth: the tilize address generator assumes one
        # contiguous [TOKENS_PER_BLOCK, FLAT] row-major stripe, so a larger CB
        # would let the ring wrap mid-block.
        CB_RM_FLAT_ROWS: flat_tiles_per_block,
        CB_FLAT_TILES: flat_tiles_per_block,
        CB_GATE_SIG: GATE_CHUNK_TILES,
    }

    # The reader/writer transfer whole DM_BLOCK_TILES groups out of one
    # get_write_ptr / get_read_ptr, so a group must never straddle the ring wrap.
    for cb_index in (CB_O_TILES, CB_GATE_TILES, CB_OUT_TILES):
        assert cb_pages[cb_index] % DM_BLOCK_TILES == 0, (
            f"onorm: CB {cb_index} has {cb_pages[cb_index]} pages, not a multiple of "
            f"DM_BLOCK_TILES={DM_BLOCK_TILES}"
        )

    total_cb_bytes = sum(cb_pages.values()) * tile_bytes
    # The CB-available slice of L1 (unreserved L1 minus the CB-region base), i.e.
    # the actual budget the knobs must fit inside.
    l1_available = ttnn.get_max_worker_l1_unreserved_size() - _L1_CB_BASE_RESERVE
    assert total_cb_bytes <= l1_available, (
        f"onorm: CB footprint {total_cb_bytes} B exceeds the CB-available L1 per core "
        f"({l1_available} B). "
        f"Lower GATE_CHUNK_TILES first (currently {GATE_CHUNK_TILES} — a pure perf/L1 "
        f"trade), then NORM_CHUNK_TOKENS (currently {NORM_CHUNK_TOKENS}). "
        f"CB_RM_FLAT_ROWS / CB_FLAT_TILES are not reducible (they are the re-tile "
        f"working set)."
    )

    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * tile_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=cb_index,
                    data_format=o.dtype,
                    page_size=tile_bytes,
                )
            ],
        )
        for cb_index, pages in cb_pages.items()
    ]

    # ================= 4. KERNELS =================
    # Reader (NCRISC / NoC0) — ALL DRAM reads live here.  Reads issued on NoC1
    # measured 4.8x slower, so the writer never reads `gate`.
    reader_ct_args = [
        v_tiles,
        TOKENS_PER_BLOCK,
        flat_tiles,
        tile_rows_per_block,
        blocks_per_batch,
        tokens,
        token_tile_rows,
        DM_BLOCK_TILES,
        _f32_bits(1.0 / head_dim),  # the reduce scaler: explicit, host-supplied 1/V
        tile_bytes,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(o).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(gate).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(weight).get_compile_time_args())

    # Writer (BRISC / NoC1) — all DRAM writes.
    writer_ct_args = [
        flat_tiles,
        tile_rows_per_block,
        blocks_per_batch,
        token_tile_rows,
        DM_BLOCK_TILES,
        tile_bytes,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    compute_ct_args = [
        NORM_CHUNK_TOKENS,
        norm_chunks_per_block,
        v_tiles,
        flat_tiles,
        tile_rows_per_block,
        GATE_CHUNK_TILES,
        gate_chunks_per_block,
    ]

    o_addr = o.buffer_address()
    gate_addr = gate.buffer_address()
    weight_addr = weight.buffer_address()
    out_addr = output.buffer_address()
    eps_bits = _f32_bits(epsilon)

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    for core, start_block, num_blocks in assignment:
        reader_rt_args[core.x][core.y] = [o_addr, gate_addr, weight_addr, start_block, num_blocks]
        writer_rt_args[core.x][core.y] = [out_addr, start_block, num_blocks]
        compute_rt_args[core.x][core.y] = [num_blocks, eps_bits]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "onorm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "onorm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "onorm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=_compute_config_descriptor(compute_kernel_config),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
