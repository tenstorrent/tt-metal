# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated micro-bench for rms_norm pass-1 (square + row-reduce), single core, sharded L1.

This reconstructs JUST the pass-1 stage of the cross-core rms_norm compute kernel
(`rms_norm_xcore_compute.cpp`, the `do_pass1` lambda) so one idea can be A/B-measured in
isolation: BATCHING the square + local reduce across the C tile-rows of a round.

Per-core pass-1 (BLOCK_SHARDED focus shape (1,1,8192,1024), 8x8 grid): each core holds a
resident W-slice of HT_LOCAL tile-rows x PER_W_T W-tiles (32 x 4 for the focus). Pass-1 produces,
per tile-row, the local partial  Sigma_slice x^2 * (1/W)  -> cb_stat_local (fp32), which the
cross-core round folds into the RMS. Everything is zero-copy sharded L1 (no DRAM, no NoC): the
measured delta is purely the on-core compute pipeline.

THE OP'S CURRENT APPROACH (honest baseline, block_rows=1): `do_pass1` loops over the C tile-rows,
and for EACH tile-row issues (1) ONE eltwise_chain square over vwt=PER_W_T tiles -> cb_xsq, then
(2) ONE reduce<SUM,REDUCE_ROW> of(1, vwt) -> one partial. Each tile-row re-pays the per-call
helper overhead (LLK init + data-format reconfig + unpack/math/pack pipeline fill/drain) on a small
(vwt=4-tile) payload. cb_xsq is bf16 depth 2*per_w_t=8 (double-buffers ONE tile-row).

THE IDEA (block_rows > 1): process `block_rows` tile-rows per pass. Batch the square across the
whole block_rows*PER_W_T tiles in ONE eltwise_chain (Block-walk), then reduce EITHER per-row
(block_rows reduces of(1,vwt), reduce_mode=0 -> isolates the SQUARE batching) OR in one blocked
reduce of(block_rows, vwt) (reduce_mode=1 -> also batches the reduce's fixed setup). This amortizes
the fixed per-call overhead over MORE tiles per call (the examples/compute_block_size mechanism).

Batching the square REQUIRES cb_xsq to hold the whole pass's square output (block_rows*PER_W_T
tiles): the square (packer) and the reduce (unpacker) are sequential helper calls on the same
compute kernel, so if cb_xsq cannot buffer the full pass the pipeline deadlocks. cb_xsq therefore
scales with block_rows -> the L1 cost that bounds the win (the predicate). Everything else is held
FIXED across variants: same math, same dtypes (bf16 in, fp32 stat out), same math_fidelity (HiFi2),
same fp32_dest_acc_en (False) -- the precision contract is never tuned.

A second lever (reconfig=off): the square reads bf16 (cb_x_in) and the reduce reads bf16 (cb_xsq),
both constant, so the square's INPUT data-format reconfig and the reduce's INPUT reconfig are
redundant. Turn them off (keeping the reduce's OUTPUT reconfig -- cb_stat_local is fp32, a real
format change -- and all inits). Correctness-gated; measures the reconfig cost + how it compounds
with block size.
"""

import struct

import ttnn

TILE = 32

# CB assignment (mirrors the op's cross-core compute kernel indices for the pass-1 CBs).
CB_X_IN = 1  # resident sharded input W-slice (bf16, zero-copy, HT_LOCAL*PER_W_T tiles)
CB_SCALER = 2  # 1/W reduce scaler (bf16, 1 tile)
CB_XSQ = 24  # x^2 pass-1 intermediate (bf16, block_rows*PER_W_T tiles -- scales with the block)
CB_STAT_LOCAL = 25  # per-tile-row local partial Sigma x^2 * (1/W) (fp32, HT_LOCAL tiles for readback)

# reduce_mode: how the block_rows*vwt squared tiles are reduced into per-row partials.
REDUCE_PER_ROW = 0  # block_rows separate reduces of(1, vwt) -- isolates the SQUARE batching only
REDUCE_BLOCKED = 1  # ONE blocked reduce of(block_rows, vwt) -- also batches the reduce's fixed setup


# =============================================================================
# Compute kernel -- ONE source; block_rows / num_blocks / reduce_mode / reconfig_on are compile-time
# args, so each variant compiles to exactly its block granularity + reduce shape + reconfig policy.
# The math (x*x per tile, REDUCE_ROW SUM * scaler per row) is IDENTICAL across every variant.
# CT args: [HT_LOCAL, PER_W_T, block_rows, num_blocks, reduce_mode, reconfig_on, kernel_iters]
# =============================================================================
_COMPUTE_KERNEL = r"""
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

// Isolated rms_norm pass-1: per tile-row, Sigma_slice x^2 * (1/W) -> cb_stat_local.
//
// block_rows > 1 batches the square across block_rows tile-rows into ONE eltwise_chain (Block-walk
// from the resident base) and (reduce_mode=1) the reduce into ONE blocked reduce of(block_rows,vwt)
// -- amortizing the fixed per-call helper overhead over more tiles per call. This is the SAME math
// as the op's per-tile-row do_pass1 (block_rows=1), only the call granularity changes.
//
// RAW-LLK NOTE: this kernel uses NO raw LLK -- every phase is a kernel_lib helper (ckl::eltwise_chain
// for the square, ckl::reduce for the row-reduce), exactly as the op does. The only "lever" is the
// block granularity fed to those helpers + the data-format reconfig flag, both of which the helpers
// expose as first-class knobs. Nothing here bypasses a helper.
void kernel_main() {
    constexpr uint32_t cb_x_in = 1;
    constexpr uint32_t cb_scaler = 2;
    constexpr uint32_t cb_xsq = 24;
    constexpr uint32_t cb_stat_local = 25;

    constexpr uint32_t HT_LOCAL = get_compile_time_arg_val(0);
    constexpr uint32_t PER_W_T = get_compile_time_arg_val(1);
    constexpr uint32_t block_rows = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t reduce_mode = get_compile_time_arg_val(4);
    constexpr bool reconfig_on = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(6);

    constexpr uint32_t shard_tiles = HT_LOCAL * PER_W_T;
    constexpr uint32_t block_tiles = block_rows * PER_W_T;

    // The square reads cb_x_in (bf16) for both operands; the reduce reads cb_xsq (bf16). Both input
    // formats are constant, so their INPUT reconfig is redundant and reconfig_on=0 turns it off. The
    // square's PACK reconfig (cb_xsq bf16) and the reduce's OUTPUT reconfig (cb_stat_local fp32 -- a
    // REAL format change) are always kept: the reduce output really does change format. Inits always
    // stay on (each phase is still a different op).
    constexpr auto SQ_IN_RC = reconfig_on ? ckl::BinaryDataFormatReconfig::Input : ckl::BinaryDataFormatReconfig::None;
    constexpr auto RD_RC = reconfig_on ? ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT
                                       : ckl::ReduceDataFormatReconfigMode::OUTPUT;

    compute_kernel_hw_startup(cb_x_in, cb_scaler, cb_stat_local);

    // Self-arm the resident zero-copy shard once (compute is its own producer; indexed access across
    // every pass, never popped) -- the op's X_ZERO_COPY path.
    cb_reserve_back(cb_x_in, shard_tiles);
    cb_push_back(cb_x_in, shard_tiles);
    cb_wait_front(cb_x_in, shard_tiles);
    cb_wait_front(cb_scaler, 1);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            const uint32_t base = blk * block_tiles;  // resident tile base of this pass's first row

            // ---- Square: block_rows*PER_W_T tiles in ONE chain (Block-walk from base) -> cb_xsq.
            // block_rows=1 degenerates to the op's per-tile-row square over PER_W_T tiles.
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(block_tiles),
                ckl::BinaryFpu<
                    cb_x_in,
                    cb_x_in,
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::None,
                    ckl::InputLifecycle::CallerManaged,
                    ckl::InputLifecycle::CallerManaged,
                    SQ_IN_RC,
                    ckl::Dst::D0,
                    ckl::OperandKind::Block,
                    ckl::OperandKind::Block,
                    ckl::TileOffset::Set,
                    ckl::TileOffset::Set>{base, base},
                ckl::PackTile<cb_xsq, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});

            // ---- Reduce: cb_xsq -> per-row partials Sigma x^2 * (1/W) in cb_stat_local.
            if constexpr (reduce_mode == 1) {
                // ONE blocked reduce of(block_rows, vwt): loops rows internally, one acquire/pack per
                // row (dst_idx 0, released per row -> no DST-budget pressure), but pays the reduce's
                // fixed setup (format reconfig + reduce_init + scaler wait) ONCE for the whole block.
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_xsq,
                    cb_scaler,
                    cb_stat_local,
                    ckl::ReduceInputPolicy::WaitAndPopPerTile,
                    RD_RC>(ckl::ReduceInputBlockShape::of(block_rows, PER_W_T, 1));
            } else {
                // Per-row reduce of(1, vwt) x block_rows -- the op's exact form (Accumulate::at iter 0
                // == NoAccumulation for ReduceTile: reload is skipped, so it is byte/perf identical to
                // a plain reduce -- kept to mirror the op faithfully). Pays the reduce setup per row.
                for (uint32_t cc = 0; cc < block_rows; ++cc) {
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_xsq,
                        cb_scaler,
                        cb_stat_local,
                        ckl::ReduceInputPolicy::WaitAndPopPerTile,
                        RD_RC>(
                        ckl::ReduceInputBlockShape::of(1, PER_W_T, 1),
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::Accumulate::at(cb_stat_local, 0),
                        ckl::NoOp{});
                }
            }
        }

        // Drain the HT_LOCAL partials between steady-state iters; leave the last pass resident for
        // readback (correctness gate). cb_stat_local is sized HT_LOCAL so a whole iter fits.
        if (iter + 1 < kernel_iters) {
            cb_wait_front(cb_stat_local, shard_tiles / PER_W_T);
            cb_pop_front(cb_stat_local, shard_tiles / PER_W_T);
        }
    }
}
"""


# =============================================================================
# Scaler kernel -- fills cb_scaler with the REDUCE_ROW SUM scaler 1/W (bf16), pushed once, never
# popped. Same pattern as examples/row_reduce_accumulate. CT args: [inv_w_bits]
# =============================================================================
_SCALER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_scaler = 2;
    constexpr uint32_t inv_w_bits = get_compile_time_arg_val(0);  // float bits of 1/W
    const float scaler = __builtin_bit_cast(float, inv_w_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        scaler);
}
"""


# =============================================================================
# Host-side sharded-L1 layout + program descriptor
# =============================================================================
def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def create_sharded_memory_config(shape):
    """`shape` (rows, cols) as a single-core height shard, tiled."""
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _scratch_cb(cb_id, data_format, num_tiles):
    tile_size = ttnn.tile_size(data_format)
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=data_format, page_size=tile_size)
    return ttnn.CBDescriptor(total_size=tile_size * num_tiles, core_ranges=_single_core(), format_descriptors=[fmt])


def valid_block_rows(block_rows, ht_local):
    return 1 <= block_rows <= ht_local and ht_local % block_rows == 0


def create_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    ht_local,
    per_w_t,
    block_rows,
    reduce_mode,
    reconfig=True,
    kernel_iters=1,
    origin_w,
):
    if input_tensor.dtype != ttnn.bfloat16 or input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("input must be bfloat16 TILE_LAYOUT")
    if output_tensor.dtype != ttnn.float32 or output_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("output must be float32 TILE_LAYOUT")
    if not valid_block_rows(block_rows, ht_local):
        raise ValueError(f"block_rows={block_rows} must divide HT_LOCAL={ht_local}")
    if reduce_mode not in (REDUCE_PER_ROW, REDUCE_BLOCKED):
        raise ValueError(f"reduce_mode must be 0 or 1, got {reduce_mode}")
    if kernel_iters < 1:
        raise ValueError("kernel_iters must be positive")

    num_blocks = ht_local // block_rows
    inv_w_bits = struct.unpack("<I", struct.pack("<f", 1.0 / float(origin_w)))[0]

    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[ht_local, per_w_t, block_rows, num_blocks, reduce_mode, int(reconfig), kernel_iters],
        # FIXED precision contract of the BLOCK_SHARDED focus case: bf16 input, HiFi2, fp32 DEST accum
        # OFF, no math_approx. Identical across every variant -- never tuned for speed.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )
    scaler = ttnn.KernelDescriptor(
        kernel_source=_SCALER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[inv_w_bits],
        runtime_args=[],
        config=ttnn.ReaderConfigDescriptor(),
    )

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X_IN, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_STAT_LOCAL, output_tensor),
        _scratch_cb(CB_SCALER, ttnn.bfloat16, 1),
        # cb_xsq holds the WHOLE pass's square output (block_rows*PER_W_T tiles) -- required to avoid a
        # same-kernel producer/consumer deadlock between the batched square (packer) and the reduce
        # (unpacker). This is the L1 cost that scales with block_rows (the predicate boundary).
        _scratch_cb(CB_XSQ, ttnn.bfloat16, block_rows * per_w_t),
    ]

    return ttnn.ProgramDescriptor(kernels=[scaler, compute], semaphores=[], cbs=cbs)


def run_op(input_tensor, *, ht_local, per_w_t, block_rows, reduce_mode, reconfig=True, kernel_iters=1, origin_w):
    """Allocate the fp32 per-row-partial output (HT_LOCAL tiles) and run one variant."""
    rows = ht_local * TILE
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows, TILE]),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        create_sharded_memory_config((rows, TILE)),
    )
    descriptor = create_program_descriptor(
        input_tensor,
        output,
        ht_local=ht_local,
        per_w_t=per_w_t,
        block_rows=block_rows,
        reduce_mode=reduce_mode,
        reconfig=reconfig,
        kernel_iters=kernel_iters,
        origin_w=origin_w,
    )
    return ttnn.generic_op([input_tensor, output], descriptor)
