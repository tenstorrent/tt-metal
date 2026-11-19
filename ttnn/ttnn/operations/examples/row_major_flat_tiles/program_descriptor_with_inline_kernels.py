# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layout::ROW_MAJOR data processed as flat tiles: skip tilize / untilize when the math does not
care where an element sits inside a tile.

Input and output are interleaved DRAM bf16 tensors [H, W] in Layout::ROW_MAJOR (one DRAM page =
one stick = one row of W elements). Two cases, each with a baseline and a lever:

    case="eltwise"  y = op(x), op = relu or gelu, applied `reps` times in DEST (compute per element knob).
        variant="tilize_untilize" (baseline): per block of 32 stick pieces (one tile-row x CH
            tile-columns), tilize -> op -> untilize.
        variant="flat_tiles": the SAME reader and writer (the same 32 stick pieces land in the
            same CB pages), and compute runs op straight on those pages, each read as a 32x32 tile.
            An elementwise op maps element i of a page to element i of the output page whatever
            the layout inside the page, so the tilize and untilize passes and their two CBs go.

    case="reduce"   y = x - mean_row(x)   (per-row centering: one reduction + one apply per row)
        variant="tilize_untilize" (baseline): per tile-row (32 sticks), tilize the whole row
            chunk by chunk into a resident tiled block, REDUCE_ROW mean over the Wt tiles,
            subtract it broadcast down the columns chunk by chunk, untilize each chunk.
        variant="flat_tiles": every stick of W = Wf x 1024 elements is Wf whole flat tiles. Per
            block of RB rows: accumulate each row's Wf flat tiles in DEST and collapse the sum to
            one scalar (REDUCE_SCALAR, x 1/W) -> one mean tile per row, then subtract that scalar
            from the row's Wf flat tiles (scalar broadcast). Only legal when W % 1024 == 0: a flat
            tile must never hold elements of two rows. Otherwise `flat_tiles` refuses (ValueError,
            see `flat_tiles_applicable`) and the baseline is the path.

Held equal between the two variants of a case: the work split (whole tile-rows of 32 sticks per
Tensix core, contiguous ranges, row-major cores), the bytes read and written, the NoC transfer
size and count (2 KB per read / write whenever W >= 1024: a 32-tile stick piece in the baseline,
one flat tile in the reduce lever) and the compute config.

Two things the reduce lever needs that the baseline does not, both measured (README, ablation):
  * one mean per ROW instead of one per tile-row: 32x the per-output reduce cost. For rows of 1-2 flat
    tiles the FPU reduce_tile datapath (cheap per output) wins; from 4 flat tiles the pairwise
    add_tiles + SFPU collapse ("accumulate") does. `resolve_reduce_algo` picks per variant; the baseline
    accumulates (its measured best at every W here).
  * its block is RB whole sticks, not 32 stick pieces: issue the 2 KB transfers flat-tile column by
    column across the block's sticks ("interleaved"), or Wf consecutive transfers go to one DRAM bank
    and the DRAM-bound cells lose 10-14% (WH B0).
"""

import math
import struct

import ttnn

TILE = 32
FLAT = TILE * TILE  # elements per flat tile
ELEM_BYTES = 2  # bf16
TILE_BYTES = FLAT * ELEM_BYTES
MAX_CHUNK_TILES = 32  # widest tile-column chunk the baseline tilizes / untilizes at once
L1_CB_BUDGET = 1024 * 1024  # bytes of CBs per Tensix core this example allows itself

CASES = ("eltwise", "reduce")
VARIANTS = ("tilize_untilize", "flat_tiles")  # baseline first
ELTWISE_OPS = ("relu", "gelu")
REDUCE_ALGOS = ("auto", "accumulate", "reduce_tile")
PIECE_ORDERS = ("interleaved", "row")
# reduce flat_tiles: rows of <= this many flat tiles use the FPU reduce_tile datapath, wider rows accumulate
# (measured crossover, WH B0: reduce_tile wins at 1-2 flat tiles per row, accumulate from 4)
REDUCE_TILE_MAX_FLAT = 2

CB_RM, CB_SCALER, CB_X, CB_MEAN, CB_Y, CB_OUT = 0, 1, 2, 3, 4, 16


# ---------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------


def chunk_tiles(wt):
    """Baseline tile-column chunk: the largest divisor of Wt that is <= MAX_CHUNK_TILES."""
    return max(d for d in range(1, min(wt, MAX_CHUNK_TILES) + 1) if wt % d == 0)


def rows_per_block(wf):
    """Reduce lever: rows per block, a power of two dividing 32 with rows x Wf <= 32 flat tiles."""
    rb = 32
    while rb > 1 and rb * wf > MAX_CHUNK_TILES:
        rb //= 2
    return rb


def flat_tiles_applicable(case, w):
    """(ok, reason): can `flat_tiles` run this case at width W?"""
    if case == "eltwise":
        return True, ""
    if w % FLAT:
        return False, (
            f"W={w} is not a multiple of {FLAT}: a flat tile (1024 contiguous elements) would hold the end "
            f"of one row and the start of the next, so a per-row reduction over whole flat tiles would mix "
            f"rows; use the tilize_untilize path"
        )
    return True, ""


def resolve_reduce_algo(variant, w, reduce_algo="auto"):
    """The reduce datapath a variant runs. "auto": the baseline accumulates (its one output per tile-row
    makes the SFPU collapse cheap); flat_tiles produces one output per ROW, 32 per tile-row, so for rows of
    <= REDUCE_TILE_MAX_FLAT flat tiles it takes the FPU reduce_tile path, whose per-output cost is lower."""
    if reduce_algo != "auto":
        return reduce_algo
    if variant == "flat_tiles" and w // FLAT <= REDUCE_TILE_MAX_FLAT:
        return "reduce_tile"
    return "accumulate"


def _baseline_x_depth(wt):
    """Reduce baseline: tiled tile-row CB depth in tiles (two tile-rows when that fits 512 KB, else one)."""
    return 2 * wt if 2 * wt * TILE_BYTES <= 512 * 1024 else wt


def cb_tiles(case, variant, w, rb=None):
    """{cb index: tiles} of the circular buffers one Tensix core allocates."""
    wt = w // TILE
    if variant == "flat_tiles" and case == "reduce":
        wf = w // FLAT
        rb = rb or rows_per_block(wf)
        return {CB_RM: 2 * rb * wf, CB_SCALER: 1, CB_MEAN: rb, CB_OUT: 2 * rb * wf}
    ch = chunk_tiles(wt)
    if variant == "flat_tiles":  # eltwise
        return {CB_RM: 2 * ch, CB_OUT: 2 * ch}
    if case == "eltwise":
        return {CB_RM: 2 * ch, CB_X: 2 * ch, CB_Y: 2 * ch, CB_OUT: 2 * ch}
    return {CB_RM: 2 * ch, CB_SCALER: 1, CB_X: _baseline_x_depth(wt), CB_MEAN: 1, CB_Y: 2 * ch, CB_OUT: 2 * ch}


def cb_bytes(case, variant, w, rb=None):
    return sum(cb_tiles(case, variant, w, rb).values()) * TILE_BYTES


# ---------------------------------------------------------------------------------------
# Kernels (inline)
# ---------------------------------------------------------------------------------------

# Reader / writer. Compile-time args:
#   0 stick_bytes (aligned DRAM page = one stick)   1 kernel_iters
#   2 mode: 0 = tile-row chunks (32 stick pieces of CH tiles, one transfer each)
#           1 = row blocks (RB whole sticks, one transfer per flat tile)
#   3 CH (tiles per chunk) or Wf (flat tiles per stick)   4 Wt or RB
#   5 piece order of mode 1: 0 = row by row, 1 = flat-tile column by column across the block's rows
#   6 reader only: reduce scaler tile, 0 = none, else the float bits of the scaler value
#   7.. TensorAccessorArgs
# Runtime args: 0 buffer address, 1 first tile-row, 2 tile-rows.
_READER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm = 0;
    constexpr uint32_t stick_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    constexpr uint32_t mode = get_compile_time_arg_val(2);
    constexpr uint32_t a = get_compile_time_arg_val(3);
    constexpr uint32_t b = get_compile_time_arg_val(4);
    constexpr uint32_t order = get_compile_time_arg_val(5);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(6);
    constexpr auto acc_args = TensorAccessorArgs<7>();
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t tr_start = get_arg_val<uint32_t>(1);
    const uint32_t tr_count = get_arg_val<uint32_t>(2);
    const auto acc = TensorAccessor(acc_args, addr, stick_bytes);

    if constexpr (scaler_bits != 0) {  // FPU reduce_tile path: scaler tile, never popped
        float scaler;
        constexpr uint32_t bits = scaler_bits;
        __builtin_memcpy(&scaler, &bits, sizeof(scaler));
        dataflow_kernel_lib::prepare_reduce_scaler<1, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(scaler);
    }
    for (uint32_t it = 0; it < kernel_iters; ++it) {
        if constexpr (mode == 0) {
            constexpr uint32_t ch = a, wt = b, piece = ch * 64;  // 32 elements x 2 B per tile-column
            for (uint32_t tr = tr_start; tr < tr_start + tr_count; ++tr) {
                for (uint32_t c = 0; c < wt; c += ch) {
                    cb_reserve_back(cb_rm, ch);
                    uint32_t l1 = get_write_ptr(cb_rm);
                    for (uint32_t s = 0; s < 32; ++s) {
                        noc_async_read(acc.get_noc_addr(tr * 32 + s, c * 64), l1, piece);
                        l1 += piece;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_rm, ch);
                }
            }
        } else {
            constexpr uint32_t wf = a, rb = b;
            for (uint32_t row = tr_start * 32; row < (tr_start + tr_count) * 32; row += rb) {
                cb_reserve_back(cb_rm, rb * wf);
                const uint32_t l1 = get_write_ptr(cb_rm);
                // CB page (r * wf + f) = flat tile f of row r, whatever the order the reads go out in
                if constexpr (order == 0) {
                    for (uint32_t r = 0; r < rb; ++r) {
                        for (uint32_t f = 0; f < wf; ++f) {
                            noc_async_read(acc.get_noc_addr(row + r, f * 2048), l1 + (r * wf + f) * 2048, 2048);
                        }
                    }
                } else {
                    for (uint32_t f = 0; f < wf; ++f) {
                        for (uint32_t r = 0; r < rb; ++r) {
                            noc_async_read(acc.get_noc_addr(row + r, f * 2048), l1 + (r * wf + f) * 2048, 2048);
                        }
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_rm, rb * wf);
            }
        }
    }
}
"""

_WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t stick_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    constexpr uint32_t mode = get_compile_time_arg_val(2);
    constexpr uint32_t a = get_compile_time_arg_val(3);
    constexpr uint32_t b = get_compile_time_arg_val(4);
    constexpr uint32_t order = get_compile_time_arg_val(5);
    constexpr auto acc_args = TensorAccessorArgs<7>();
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t tr_start = get_arg_val<uint32_t>(1);
    const uint32_t tr_count = get_arg_val<uint32_t>(2);
    const auto acc = TensorAccessor(acc_args, addr, stick_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        if constexpr (mode == 0) {
            constexpr uint32_t ch = a, wt = b, piece = ch * 64;
            for (uint32_t tr = tr_start; tr < tr_start + tr_count; ++tr) {
                for (uint32_t c = 0; c < wt; c += ch) {
                    cb_wait_front(cb_out, ch);
                    uint32_t l1 = get_read_ptr(cb_out);
                    for (uint32_t s = 0; s < 32; ++s) {
                        noc_async_write(l1, acc.get_noc_addr(tr * 32 + s, c * 64), piece);
                        l1 += piece;
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_out, ch);
                }
            }
        } else {
            constexpr uint32_t wf = a, rb = b;
            for (uint32_t row = tr_start * 32; row < (tr_start + tr_count) * 32; row += rb) {
                cb_wait_front(cb_out, rb * wf);
                const uint32_t l1 = get_read_ptr(cb_out);
                if constexpr (order == 0) {
                    for (uint32_t r = 0; r < rb; ++r) {
                        for (uint32_t f = 0; f < wf; ++f) {
                            noc_async_write(l1 + (r * wf + f) * 2048, acc.get_noc_addr(row + r, f * 2048), 2048);
                        }
                    }
                } else {
                    for (uint32_t f = 0; f < wf; ++f) {
                        for (uint32_t r = 0; r < rb; ++r) {
                            noc_async_write(l1 + (r * wf + f) * 2048, acc.get_noc_addr(row + r, f * 2048), 2048);
                        }
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out, rb * wf);
            }
        }
    }
}
"""

# Elementwise compute. Compile-time args: 0 CH, 1 Wt, 2 kernel_iters, 3 flat (0 / 1). __OPS__ / __REPS__ are
# substituted on the host: the op element (Relu<>{} or Gelu<>{}) repeated `reps` times.
# Runtime arg: 0 tile-rows.
_ELTWISE_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"

using namespace compute_kernel_lib;

// One chain: copy the tile to DEST, run the op __REPS__ time(s) in DEST, pack.
template <uint32_t n, uint32_t cb_a, uint32_t cb_b>
ALWI void apply_op(DataflowBuffer& a, DataflowBuffer& b) {
    eltwise_chain(
        IterationShape::tiles(n).block_size(8),
        CopyTile<input(cb_a, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block), Dst::D0>{a}
        __OPS__,
        PackTile<output(cb_b, ReservePolicy::Upfront, PushPolicy::AtEnd)>{b});
}

void kernel_main() {
    constexpr uint32_t cb_rm = 0, cb_x = 2, cb_y = 4, cb_out = 16;
    constexpr uint32_t ch = get_compile_time_arg_val(0);
    constexpr uint32_t wt = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);
    constexpr bool flat = get_compile_time_arg_val(3) != 0;
    const uint32_t tile_rows = get_arg_val<uint32_t>(0);

    DataflowBuffer rm(cb_rm), x(cb_x), y(cb_y), out(cb_out);
    compute_kernel_hw_startup(cb_rm, cb_out);
    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t tr = 0; tr < tile_rows; ++tr) {
            for (uint32_t c = 0; c < wt; c += ch) {
                if constexpr (flat) {
                    // 32 stick pieces = ch pages; each page is read as a 32x32 tile, op'd, packed back.
                    apply_op<ch, cb_rm, cb_out>(rm, out);
                } else {
                    tilize<ch, cb_rm, cb_x>(1);
                    apply_op<ch, cb_x, cb_y>(x, y);
                    untilize<ch, cb_y, cb_out>(1);
                }
            }
        }
    }
}
"""

# Reduce baseline compute. Compile-time args: 0 CH, 1 Wt, 2 kernel_iters, 3 W (elements per row),
# 4 reduce datapath (0 accumulate, 1 reduce_tile).
_REDUCE_BASELINE_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

using namespace compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_rm = 0, cb_scaler = 1, cb_x = 2, cb_mean = 3, cb_y = 4, cb_out = 16;
    constexpr uint32_t ch = get_compile_time_arg_val(0);
    constexpr uint32_t wt = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);
    constexpr uint32_t w = get_compile_time_arg_val(3);
    constexpr uint32_t algo = get_compile_time_arg_val(4);
    const uint32_t tile_rows = get_arg_val<uint32_t>(0);

    DataflowBuffer x(cb_x), mean(cb_mean), y(cb_y);
    compute_kernel_hw_startup(cb_rm, cb_out);
    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t tr = 0; tr < tile_rows; ++tr) {
            // 1. tilize the tile-row (32 sticks) chunk by chunk into the resident tiled block
            for (uint32_t c = 0; c < wt; c += ch) {
                tilize<ch, cb_rm, cb_x>(1);
            }
            // 2. mean of each of the 32 rows: Wt tiles -> column 0 of one tile
            if constexpr (algo == 0) {  // pairwise add_tiles in DEST + SFPU collapse + x 1/W
                reduce_mean<ReduceDim::REDUCE_ROW, cb_x, cb_scaler, cb_mean, ReduceInputPolicy::WaitUpfrontNoPop>(
                    ReduceInputBlockShape::of(1, wt, 1), w);
            } else {  // FPU reduce_tile per tile, scaler tile = 1/W
                reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_x, cb_scaler, cb_mean, ReduceInputPolicy::WaitUpfrontNoPop>(
                    ReduceInputBlockShape::of(1, wt, 1));
            }
            // 3. y = x - mean (broadcast down the columns), 4. untilize, chunk by chunk
            for (uint32_t c = 0; c < wt; c += ch) {
                sub<input(cb_x, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
                    input(cb_mean, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Scalar),
                    output(cb_y, ReservePolicy::Upfront, PushPolicy::AtEnd)>(
                    IterationShape::tiles(ch).block_size(8), x, mean, y);
                untilize<ch, cb_y, cb_out>(1);
            }
            mean.pop_front(1);
        }
    }
}
"""

# Reduce lever compute. Compile-time args: 0 Wf (flat tiles per row), 1 RB (rows per block),
# 2 kernel_iters, 3 W, 4 reduce datapath (0 accumulate, 1 reduce_tile). Runtime arg: 0 tile-rows.
_REDUCE_FLAT_COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

using namespace compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_rm = 0, cb_scaler = 1, cb_mean = 3, cb_out = 16;
    constexpr uint32_t wf = get_compile_time_arg_val(0);
    constexpr uint32_t rb = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);
    constexpr uint32_t w = get_compile_time_arg_val(3);
    constexpr uint32_t algo = get_compile_time_arg_val(4);
    const uint32_t tile_rows = get_arg_val<uint32_t>(0);
    const uint32_t blocks = tile_rows * (32 / rb);

    DataflowBuffer rm(cb_rm), mean(cb_mean), out(cb_out);
    compute_kernel_hw_startup(cb_rm, cb_out);
    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t b = 0; b < blocks; ++b) {
            // 1. per row (batch of Wf flat tiles): pairwise add the Wf tiles in DEST, collapse the whole
            //    tile to one scalar, x 1/W -> one mean tile per row (value at [0, 0])
            if constexpr (algo == 0) {  // pairwise add_tiles in DEST + SFPU collapse + x 1/W
                reduce_mean<ReduceDim::REDUCE_SCALAR, cb_rm, cb_scaler, cb_mean, ReduceInputPolicy::WaitUpfrontNoPop>(
                    ReduceInputBlockShape::of(1, wf, rb), w);
            } else {  // FPU reduce_tile per flat tile; the scaler (1/sqrt(W)) is applied twice: rows, then columns
                reduce<PoolType::SUM, ReduceDim::REDUCE_SCALAR, cb_rm, cb_scaler, cb_mean, ReduceInputPolicy::WaitUpfrontNoPop>(
                    ReduceInputBlockShape::of(1, wf, rb));
            }
            // 2. y = x - mean: row r's Wf flat tiles minus mean tile r, scalar broadcast
            sub<input(cb_rm, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
                input(cb_mean, BroadcastDim::Scalar, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
                output(cb_out, ReservePolicy::Upfront, PushPolicy::AtEnd)>(
                IterationShape::grid(rb, wf).block_size(8), rm, mean, out);
        }
    }
}
"""


# ---------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------


def _check_input(input_tensor):
    shape = list(input_tensor.shape)
    if len(shape) != 2:
        raise ValueError(f"row_major_flat_tiles: rank must be 2, got {len(shape)}")
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT or input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("row_major_flat_tiles: input must be ROW_MAJOR bfloat16")
    mc = input_tensor.memory_config()
    if mc.is_sharded() or mc.buffer_type != ttnn.BufferType.DRAM:
        raise ValueError("row_major_flat_tiles: input must be DRAM interleaved")
    h, w = shape
    if h % TILE or w % TILE:
        raise ValueError(f"row_major_flat_tiles: H and W must be multiples of {TILE}, got {shape}")
    return h, w


def work_split(device, tile_rows):
    """[(core, first tile-row, tile-rows)]: contiguous tile-row ranges over min(tile-rows, grid) Tensix
    cores, row-major. Identical for both variants of a case."""
    grid = device.compute_with_storage_grid_size()
    n = max(1, min(tile_rows, grid.x * grid.y))
    base, rem = divmod(tile_rows, n)
    out, start = [], 0
    for k in range(n):
        count = base + (1 if k < rem else 0)
        out.append((ttnn.CoreCoord(k % grid.x, k // grid.x), start, count))
        start += count
    return out


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def create_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    case,
    variant,
    op="relu",
    reps=1,
    reduce_algo=REDUCE_ALGOS[0],
    piece_order=PIECE_ORDERS[0],
    rows_per_block_override=None,
    kernel_iters=1,
):
    """See `row_major_flat_tiles` for the arguments."""
    if case not in CASES:
        raise ValueError(f"row_major_flat_tiles: case must be one of {CASES}, got {case!r}")
    if variant not in VARIANTS:
        raise ValueError(f"row_major_flat_tiles: variant must be one of {VARIANTS}, got {variant!r}")
    if op not in ELTWISE_OPS or reps < 1:
        raise ValueError(f"row_major_flat_tiles: op must be one of {ELTWISE_OPS} and reps >= 1, got {op!r} x {reps}")
    if reduce_algo not in REDUCE_ALGOS or piece_order not in PIECE_ORDERS:
        raise ValueError(f"row_major_flat_tiles: reduce_algo in {REDUCE_ALGOS}, piece_order in {PIECE_ORDERS}")
    h, w = _check_input(input_tensor)
    if variant == "flat_tiles":
        ok, reason = flat_tiles_applicable(case, w)
        if not ok:
            raise ValueError(f"row_major_flat_tiles: flat_tiles does not apply: {reason}")
    rb = rows_per_block_override
    if case == "reduce" and variant == "flat_tiles" and rb is not None and (rb < 1 or 32 % rb):
        raise ValueError(f"row_major_flat_tiles: rows per block must divide 32, got {rb}")
    footprint = cb_bytes(case, variant, w, rb)
    if footprint > L1_CB_BUDGET:
        raise ValueError(f"row_major_flat_tiles: {footprint} B of CBs per Tensix core for W={w} exceed {L1_CB_BUDGET}")

    device = input_tensor.device()
    wt = w // TILE
    split = work_split(device, h // TILE)
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in split])
    stick_bytes = input_tensor.buffer_aligned_page_size()
    reduce_algo = resolve_reduce_algo(variant, w, reduce_algo)
    algo = ("accumulate", "reduce_tile").index(reduce_algo)

    scaler = 0
    if case == "eltwise":
        ch = chunk_tiles(wt)
        dm_ct = [stick_bytes, kernel_iters, 0, ch, wt, 0]
        op_element = {"relu": "Relu<>{}", "gelu": "Gelu<>{}"}[op]
        compute_src = _ELTWISE_COMPUTE.replace("__OPS__", (",\n        " + op_element) * reps).replace(
            "__REPS__", str(reps)
        )
        compute_ct = [ch, wt, kernel_iters, int(variant == "flat_tiles")]
    elif variant == "flat_tiles":
        wf = w // FLAT
        rb = rb or rows_per_block(wf)
        dm_ct = [stick_bytes, kernel_iters, 1, wf, rb, int(piece_order == "interleaved")]
        compute_src, compute_ct = _REDUCE_FLAT_COMPUTE, [wf, rb, kernel_iters, w, algo]
        if reduce_algo == "reduce_tile":
            scaler = _float_bits(1.0 / math.sqrt(w))  # applied twice by a REDUCE_SCALAR reduce_tile
    else:
        ch = chunk_tiles(wt)
        dm_ct = [stick_bytes, kernel_iters, 0, ch, wt, 0]
        compute_src, compute_ct = _REDUCE_BASELINE_COMPUTE, [ch, wt, kernel_iters, w, algo]
        if reduce_algo == "reduce_tile":
            scaler = _float_bits(1.0 / w)

    cbs = [
        ttnn.CBDescriptor(
            total_size=tiles * TILE_BYTES,
            core_ranges=core_ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=TILE_BYTES)
            ],
        )
        for index, tiles in cb_tiles(case, variant, w, rb).items()
    ]

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    in_addr, out_addr = input_tensor.buffer_address(), output_tensor.buffer_address()
    for core, start, count in split:
        reader_rt[core.x][core.y] = [in_addr, start, count]
        writer_rt[core.x][core.y] = [out_addr, start, count]
        compute_rt[core.x][core.y] = [count]

    if case == "reduce":
        # identical for both variants: fp32 DEST for the row sums, HiFi4
        compute_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    else:
        compute_cfg = ttnn.ComputeConfigDescriptor()

    def kernel(src, ct_args, rt, config):
        return ttnn.KernelDescriptor(
            kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=core_ranges,
            compile_time_args=ct_args,
            runtime_args=rt,
            config=config,
        )

    reader_ct = dm_ct + [scaler] + list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct = dm_ct + [0] + list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    return ttnn.ProgramDescriptor(
        kernels=[
            kernel(_READER, reader_ct, reader_rt, ttnn.ReaderConfigDescriptor()),
            kernel(_WRITER, writer_ct, writer_rt, ttnn.WriterConfigDescriptor()),
            kernel(compute_src, compute_ct, compute_rt, compute_cfg),
        ],
        semaphores=[],
        cbs=cbs,
    )


def row_major_flat_tiles(
    input_tensor,
    *,
    case="eltwise",
    variant="flat_tiles",
    op="relu",
    reps=1,
    reduce_algo=REDUCE_ALGOS[0],
    piece_order=PIECE_ORDERS[0],
    rows_per_block_override=None,
    kernel_iters=1,
):
    """case="eltwise": y = op(x); case="reduce": y = x - mean_row(x). Input and output: interleaved DRAM
    ROW_MAJOR bf16 [H, W], H and W multiples of 32 (reduce + flat_tiles: W a multiple of 1024).

    variant: "tilize_untilize" (baseline) or "flat_tiles" (no tilize / untilize).
    op, reps (eltwise): "relu" or "gelu", applied `reps` times in DEST per tile (compute per element knob).
    reduce_algo (reduce): "accumulate" (pairwise add_tiles in DEST + SFPU within-tile collapse),
        "reduce_tile" (FPU reduce_tile per tile with a scaler tile) or "auto" (`resolve_reduce_algo`:
        each variant's measured-fastest datapath).
    piece_order (reduce flat_tiles): order of the block's 2 KB reads / writes - "interleaved" (flat-tile
        column by column across the block's sticks, so consecutive transfers hit different DRAM banks) or
        "row" (one stick after the other: Wf consecutive transfers to one bank).
    rows_per_block_override (reduce flat_tiles): rows per block (divides 32); default `rows_per_block`.
    kernel_iters: in-kernel repeat (1 = per-launch latency)."""
    if kernel_iters < 1:
        raise ValueError(f"row_major_flat_tiles: kernel_iters must be >= 1, got {kernel_iters}")
    _check_input(input_tensor)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        input_tensor.device(),
        ttnn.DRAM_MEMORY_CONFIG,
    )
    descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        case=case,
        variant=variant,
        op=op,
        reps=reps,
        reduce_algo=reduce_algo,
        piece_order=piece_order,
        rows_per_block_override=rows_per_block_override,
        kernel_iters=kernel_iters,
    )
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)


__all__ = [
    "CASES",
    "ELTWISE_OPS",
    "PIECE_ORDERS",
    "REDUCE_ALGOS",
    "VARIANTS",
    "cb_bytes",
    "chunk_tiles",
    "create_program_descriptor",
    "flat_tiles_applicable",
    "resolve_reduce_algo",
    "row_major_flat_tiles",
    "rows_per_block",
    "work_split",
]
