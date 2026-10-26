# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multi-core data-movement benchmark: read only the bytes of a tile the kernel uses.

A row vector broadcast over rows — a bias, a scale, a per-column weight — is stored as a
`Layout::TILE` tensor of shape [32, W] in DRAM (`TensorMemoryLayout::INTERLEAVED`). Only row 0 of
every 32x32 tile is meaningful; rows 1-31 are padding the tile format carries. Every Tensix core
needs the whole vector (W/32 tiles), and each Tensix core adds it to its own L1-resident input
(`rows` tile-rows of W/32 tiles, height-sharded across Tensix cores):

    out[r, c] = x[r, c] + v[0, c]          (add_tiles_bcast_rows: srcB row 0 broadcast over rows)

A bf16 32x32 tile is 2048 B laid out as four 16x16 faces of 512 B each. Row 0 of the tile is the
first 32 B of face 0 (columns 0-15) and the first 32 B of face 1 (columns 16-31). The broadcast
unpacker reads only those two face-rows.

Variants (only the reader differs — CBs, compute, the work split and the output are identical):
  whole_tile   (baseline) — one `noc_async_read_page` per tile: 2048 B per tile.
  row0_only    (lever)    — two `noc_async_read`s per tile, one at face 0 row 0 and one at face 1
                            row 0, each `seg_bytes` long: the 32 B of data rounded up to the DRAM
                            read alignment (64 B on Blackhole, 32 B on Wormhole — queried from the
                            device, never hard-coded). Each segment lands at the same byte offset
                            inside the CB tile slot as in a whole-tile read, so the consumer is
                            unchanged. 2 x seg_bytes per tile: 128 B (BH) / 64 B (WH) instead of 2048 B.
  row0_span               — the same bytes in ONE read: from the start of face 0 through face 1's
                            first segment (512 + seg_bytes = 576 B on BH). Half the requests of
                            row0_only, 4.5x more bytes: separates "bound by bytes" from "bound by
                            requests".

Stage ablation `no_read` (not a method): the reader pushes the CB pages without reading, so its time
is everything except the vector read.

All readers batch `block` tiles per `noc_async_read_barrier` into a CB of depth 2 x block, so the
baseline is the honest, already-pipelined reader — the only difference is bytes per tile.
"""

import math

import ttnn

TILE = 32
FACE = 16

CB_X = 0  # resident input, height-sharded in L1 (rows * Wt tiles per Tensix core)
CB_V = 1  # the streamed row-broadcast operand (2 x block tiles)
CB_OUT = 16  # output, height-sharded in L1 (rows * Wt tiles per Tensix core)

VARIANTS = ("whole_tile", "row0_only", "row0_span")  # baseline first
BASELINE = "whole_tile"
LEVER = "row0_only"
# Stage ablation, not a method: the reader pushes the CB pages without reading anything, so the
# measured time is everything except the vector read (output is garbage and is never checked).
ABLATION = "no_read"
_READ_MODE = {"whole_tile": 0, "row0_only": 1, "no_read": 2, "row0_span": 3}

DEFAULT_BLOCK = 8  # tiles per read barrier (both variants)
DST_BATCH = 4  # tiles per tile_regs_acquire in compute (fits half-sync DST for bf16 on both archs)

# =============================================================================
# Reader (NCRISC) — marks the sharded input resident, then streams the vector from DRAM.
# CT: [mode, wt, x_tiles, block, seg_bytes, face_bytes, iters, *TensorAccessorArgs(v)]   RT: [v_addr]
# =============================================================================
_READER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_x = 0, cb_v = 1;
    constexpr uint32_t mode       = get_compile_time_arg_val(0);  // 0 whole, 1 row 0 (2 reads), 2 none, 3 span
    constexpr uint32_t wt         = get_compile_time_arg_val(1);  // vector length in tiles
    constexpr uint32_t x_tiles    = get_compile_time_arg_val(2);  // resident input tiles on this Tensix core
    constexpr uint32_t block      = get_compile_time_arg_val(3);  // tiles per read barrier (divides wt)
    constexpr uint32_t seg_bytes  = get_compile_time_arg_val(4);  // bytes per face-row read (DRAM-aligned)
    constexpr uint32_t face_bytes = get_compile_time_arg_val(5);  // byte offset of face 1 inside a tile
    constexpr uint32_t iters      = get_compile_time_arg_val(6);  // in-kernel repeats of the whole stream
    constexpr auto v_args = TensorAccessorArgs<7>();
    static_assert(wt % block == 0, "block must divide the vector length in tiles");
    static_assert(seg_bytes % NOC_DRAM_READ_ALIGNMENT_BYTES == 0, "segment must respect the DRAM read alignment");
    static_assert(face_bytes % NOC_DRAM_READ_ALIGNMENT_BYTES == 0, "face 1 must start DRAM-read aligned");

    const uint32_t v_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_bytes = get_tile_size(cb_v);
    const auto v = TensorAccessor(v_args, v_addr, tile_bytes);

    cb_reserve_back(cb_x, x_tiles);  // sharded input is already in L1: publish it once
    cb_push_back(cb_x, x_tiles);

    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t k0 = 0; k0 < wt; k0 += block) {
            cb_reserve_back(cb_v, block);
            uint32_t l1 = get_write_ptr(cb_v);
            for (uint32_t i = 0; i < block; ++i) {
                if constexpr (mode == 1) {
                    // Row 0 = first bytes of face 0 (cols 0-15) and of face 1 (cols 16-31); each lands at
                    // the same offset inside the tile slot, so the broadcast unpacker sees a normal tile.
                    noc_async_read(v.get_noc_addr(k0 + i, 0), l1, seg_bytes);
                    noc_async_read(v.get_noc_addr(k0 + i, face_bytes), l1 + face_bytes, seg_bytes);
                } else if constexpr (mode == 3) {
                    // One read from the start of face 0 to the end of face 1's first segment.
                    noc_async_read(v.get_noc_addr(k0 + i, 0), l1, face_bytes + seg_bytes);
                } else if constexpr (mode == 0) {
                    noc_async_read_page(k0 + i, v, l1);
                }
                l1 += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_v, block);
        }
    }
}
"""

# =============================================================================
# Compute — out[r, k] = x[r, k] + bcast_row(v[k]) for every resident tile-row r; packs straight into
# the sharded output (out-of-order pack, one reserve / push per launch).
# CT: [wt, rows, block, iters]
# =============================================================================
_COMPUTE_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = 0, cb_v = 1, cb_out = 16;
    constexpr uint32_t wt    = get_compile_time_arg_val(0);
    constexpr uint32_t rows  = get_compile_time_arg_val(1);
    constexpr uint32_t block = get_compile_time_arg_val(2);
    constexpr uint32_t iters = get_compile_time_arg_val(3);
    constexpr uint32_t dst_batch = 4;
    constexpr uint32_t x_tiles = rows * wt;

    compute_kernel_hw_startup(cb_x, cb_v, cb_out);
    add_bcast_rows_init(cb_x, cb_v);

    cb_wait_front(cb_x, x_tiles);
    cb_reserve_back(cb_out, x_tiles);
    uint32_t out_idx[dst_batch];

    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t k0 = 0; k0 < wt; k0 += block) {
            cb_wait_front(cb_v, block);
            uint32_t j = 0;
            for (uint32_t i = 0; i < block; ++i) {
                for (uint32_t r = 0; r < rows; ++r) {
                    const uint32_t t = r * wt + k0 + i;
                    if (j == 0) { tile_regs_acquire(); }
                    add_tiles_bcast_rows(cb_x, cb_v, t, i, j);
                    out_idx[j++] = t;
                    if (j == dst_batch || (i == block - 1 && r == rows - 1)) {
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t q = 0; q < j; ++q) { pack_tile<true>(q, cb_out, out_idx[q]); }
                        tile_regs_release();
                        j = 0;
                    }
                }
            }
            cb_pop_front(cb_v, block);
        }
    }
    cb_push_back(cb_out, x_tiles);
    cb_pop_front(cb_x, x_tiles);
}
"""


# =============================================================================
# Host side
# =============================================================================
def segment_bytes(device, dtype=ttnn.bfloat16):
    """Bytes per face-row read: one face row of data rounded up to the DRAM read alignment."""
    elem = {ttnn.bfloat16: 2}[dtype]
    align = ttnn.get_dram_alignment()
    return math.ceil(FACE * elem / align) * align


def bytes_per_tile(variant, device, tile_bytes):
    """DRAM bytes one Tensix core reads for one vector tile."""
    seg = segment_bytes(device)
    return {BASELINE: tile_bytes, LEVER: 2 * seg, "row0_span": tile_bytes // 4 + seg, ABLATION: 0}[variant]


def core_grid(device, num_cores):
    grid = device.compute_with_storage_grid_size()
    if not 1 <= num_cores <= grid.x * grid.y:
        raise ValueError(f"num_cores must be in [1, {grid.x * grid.y}], got {num_cores}")
    return ttnn.num_cores_to_corerangeset(num_cores, grid, True)


def create_sharded_memory_config(device, num_cores, rows, width):
    """Height-shard a [num_cores * rows * 32, width] tensor: `rows` tile-rows per Tensix core."""
    return ttnn.create_sharded_memory_config(
        shape=(rows * TILE, width),
        core_grid=core_grid(device, num_cores),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def effective_block(wt, block):
    """Largest divisor of wt that is <= block (the CB holds 2 x block tiles; no partial blocks)."""
    return math.gcd(wt, block) if block < wt else wt


def create_program_descriptor(x, vec, out, *, variant, num_cores, rows, block=DEFAULT_BLOCK, kernel_iters=1):
    """x / out: [num_cores * rows * 32, W] bf16 TILE, height-sharded in L1, `rows` tile-rows per
    Tensix core. vec: [32, W] bf16 TILE in DRAM (interleaved); only its row 0 is used."""
    if variant not in _READ_MODE:
        raise ValueError(f"variant must be one of {VARIANTS + (ABLATION,)}, got {variant!r}")
    for name, t in (("x", x), ("vec", vec), ("out", out)):
        if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"{name} must be bfloat16 TILE_LAYOUT")
    if vec.memory_config().buffer_type != ttnn.BufferType.DRAM or vec.memory_config().is_sharded():
        raise ValueError("vec must be DRAM interleaved")
    v_shape = list(vec.shape)
    if len(v_shape) != 2 or v_shape[0] != TILE or v_shape[1] % TILE:
        raise ValueError(f"vec must be [32, W] with W % 32 == 0, got {v_shape}")
    width = v_shape[1]
    if list(x.shape) != [num_cores * rows * TILE, width]:
        raise ValueError(f"x must be [{num_cores * rows * TILE}, {width}], got {list(x.shape)}")
    if kernel_iters < 1 or rows < 1 or block < 1:
        raise ValueError("kernel_iters, rows and block must be positive")

    device = vec.device()
    wt = width // TILE
    blk = effective_block(wt, block)
    tile_bytes = vec.buffer_aligned_page_size()
    face_bytes = tile_bytes // 4  # bf16 32x32: four 16x16 faces
    seg = segment_bytes(device)
    crs = core_grid(device, num_cores)
    cores = ttnn.corerange_to_cores(crs, row_wise=True)

    cb_v = ttnn.CBDescriptor(
        total_size=2 * blk * tile_bytes,
        core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_V, data_format=vec.dtype, page_size=tile_bytes)],
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, x),
        cb_v,
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
    ]

    reader_rt = ttnn.RuntimeArgs()
    for c in cores:
        reader_rt[c.x][c.y] = [vec.buffer_address()]
    reader = ttnn.KernelDescriptor(
        kernel_source=_READER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=crs,
        compile_time_args=[
            _READ_MODE[variant],
            wt,
            rows * wt,
            blk,
            seg,
            face_bytes,
            kernel_iters,
            *ttnn.TensorAccessorArgs(vec).get_compile_time_args(),
        ],
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=crs,
        compile_time_args=[wt, rows, blk, kernel_iters],
        config=ttnn.ComputeConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, compute], semaphores=[], cbs=cbs)


def partial_tile_read(x, vec, *, variant, num_cores, rows, block=DEFAULT_BLOCK, kernel_iters=1):
    """out = x + broadcast(vec row 0) over rows; every Tensix core reads the whole vector from DRAM."""
    out = ttnn.allocate_tensor_on_device(x.shape, x.dtype, x.layout, x.device(), x.memory_config())
    descriptor = create_program_descriptor(
        x, vec, out, variant=variant, num_cores=num_cores, rows=rows, block=block, kernel_iters=kernel_iters
    )
    return ttnn.generic_op([x, vec, out], descriptor)
