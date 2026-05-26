// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Softmax compute kernel — Refinement 1 (chunked, L1-budget-bounded) +
// Refinement 4 (partial-scaler routing for non-tile-aligned reduce dim).
//
// Computes per-strip numerically-stable softmax in three passes over the input,
// keeping per-core L1 CB usage bounded by a constant `BLOCK_SIZE` instead of
// `reduce_dim_tiles`. The reader streams `x` `num_input_passes` times.
//
//   numeric_stable = True (default, 3 reader passes / 3 compute passes):
//     Pass 1 (MAX):
//       reduce<MAX, REDUCE_DIM, WaitAndPopPerTile>(cb_input_tiles, cb_max_scaler,
//                                                  cb_max, shape=(1,Wt) or (Ht,1))
//       The reduce holds DST across all `reduce_dim_tiles` tiles internally and
//       packs once at the end, so cb_input_tiles only needs to hold 1 tile at a
//       time (we double-buffer to 2 for reader-pipelining headroom).
//
//     Pass 2 (SUM(exp(x - max))):
//       for (b = 0..NUM_BLOCKS-1):
//         sub<BCAST, WaitAndPopPerTile, WaitUpfrontNoPop>(
//             cb_input_tiles, cb_max, cb_centered_exp, BLOCK_shape, exp_postop)
//         accumulate_reduce_block<SUM, REDUCE_DIM>(
//             cb_centered_exp, cb_sum_scaler, cb_inv_sum,
//             BLOCK_shape, b, NUM_BLOCKS, partial=none, recip_postop)
//       The wrapper applies the recip postop only on the LAST block, so
//       cb_inv_sum ends Pass 2 holding `1 / Σ exp(x - max)`.
//
//     Pass 3 (output = exp(x - max) * inv_sum):
//       for (b = 0..NUM_BLOCKS-1):
//         sub<BCAST, WaitAndPopPerTile, WaitUpfrontNoPop>(
//             cb_input_tiles, cb_max, cb_centered_exp, BLOCK_shape, exp_postop)
//         mul<BCAST, WaitAndPopPerTile, WaitUpfrontNoPop>(
//             cb_centered_exp, cb_inv_sum, cb_output_tiles, BLOCK_shape)
//       After Pass 3, pop cb_max (1) and cb_inv_sum (1) — the persistent CBs
//       used across passes.
//
//   numeric_stable = False (2 reader passes / 2 compute passes):
//     Pass 1 (SUM(exp(x))):
//       for (b = 0..NUM_BLOCKS-1):
//         sfpu_exp<cb_input_tiles>(cb_centered_exp, BLOCK_SIZE)
//         accumulate_reduce_block<SUM, REDUCE_DIM>(
//             cb_centered_exp, cb_sum_scaler, cb_inv_sum,
//             BLOCK_shape, b, NUM_BLOCKS, partial=none, recip_postop)
//     Pass 2 (output = exp(x) * inv_sum):
//       for (b = 0..NUM_BLOCKS-1):
//         sfpu_exp<cb_input_tiles>(cb_centered_exp, BLOCK_SIZE)
//         mul<BCAST, WaitAndPopPerTile, WaitUpfrontNoPop>(
//             cb_centered_exp, cb_inv_sum, cb_output_tiles, BLOCK_shape)
//
// MAX + REDUCE_ROW + Accumulate::at is forbidden by the LLK (see
// reduce_helpers_compute.inl:181 — the pack-reduce edge mask drops the running
// accumulator on the reload pass). We sidestep the limitation by streaming the
// full reduce dim through a single reduce<MAX, WaitAndPopPerTile> call.
//
// CB sync (per strip, numeric_stable=True):
//   cb_input_tiles : reader pushes 3 × reduce_dim_tiles tiles (per-tile);
//                    compute pops 3 × reduce_dim_tiles tiles (per-tile, via the
//                    reduce/sub/sub helpers' WaitAndPopPerTile policy).
//   cb_max         : Pass 1 pushes 1 (held by Pass 2 + Pass 3); after Pass 3
//                    the kernel pops 1.
//   cb_inv_sum     : Pass 2 pushes 1 (held by Pass 3); after Pass 3 the kernel
//                    pops 1.
//   cb_centered_exp: per block: sub pushes BLOCK_SIZE, reduce/mul pops BLOCK_SIZE.
//                    Net balanced after each block.
//   cb_output_tiles: per block (Pass 3): mul pushes BLOCK_SIZE; writer drains.
//                    NUM_BLOCKS × BLOCK_SIZE = reduce_dim_tiles, matching writer.
//   Scalers        : one-shot at boot, NoPop, persistent.
//
// `fp32_dest_acc_en=True` halves DEST capacity to 4 tiles. All helpers honor
// `DEST_AUTO_LIMIT` automatically; the kernel never hand-codes a DEST loop.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"

#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"

namespace {
constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_max_scaler = 8;
constexpr uint32_t cb_sum_scaler = 9;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_max = 24;
constexpr uint32_t cb_inv_sum = 25;
constexpr uint32_t cb_centered_exp = 26;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t dim_is_row = get_compile_time_arg_val(0);
    constexpr uint32_t numeric_stable_flag = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t reduce_dim_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(5);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(6);
    constexpr uint32_t has_partial = get_compile_time_arg_val(7);  // Refinement 4
    (void)Ht;
    (void)Wt;
    (void)reduce_dim_tiles;

    const uint32_t num_strips = get_arg_val<uint32_t>(0);

    // ---- One-shot hardware startup (must come before any helper) ----
    compute_kernel_hw_startup(cb_input_tiles, cb_max_scaler, cb_output_tiles);

    // ---- Dim-dependent helper shape and template choices ----
    // Full-strip shape for Pass 1 (MAX over the full reduce dim).
    constexpr auto pass1_reduce_shape =
        (dim_is_row != 0) ? ckl::ReduceInputBlockShape::of(1, Wt) : ckl::ReduceInputBlockShape::of(Ht, 1);

    // Per-block shape for Passes 2 and 3.
    constexpr auto block_reduce_shape = (dim_is_row != 0) ? ckl::ReduceInputBlockShape::of(1, BLOCK_SIZE)
                                                          : ckl::ReduceInputBlockShape::of(BLOCK_SIZE, 1);

    constexpr auto block_bin_shape = (dim_is_row != 0) ? ckl::BinaryInputBlockShape::of(1, BLOCK_SIZE)
                                                       : ckl::BinaryInputBlockShape::of(BLOCK_SIZE, 1);

    constexpr auto REDUCE_AXIS = (dim_is_row != 0) ? ckernel::ReduceDim::REDUCE_ROW : ckernel::ReduceDim::REDUCE_COL;

    constexpr auto BCAST_AXIS = (dim_is_row != 0) ? ckl::BroadcastDim::COL : ckl::BroadcastDim::ROW;

    constexpr bool NUMERIC_STABLE = (numeric_stable_flag != 0);

    // Refinement 4: partial-scaler selector. When the reduce-dim's logical size
    // is not a multiple of 32, the reader pushed a (full, partial) scaler tile
    // pair into both scaler CBs; the helper picks tile 1 for the last
    // reduce-dim iteration. Aligned shapes pass `none()`, which keeps the
    // single-tile behaviour.
    constexpr auto partial_scaler =
        (has_partial != 0) ? ckl::ReducePartialScaler::last_tile_at(1) : ckl::ReducePartialScaler::none();

    // ---- Strip loop ----
    for (uint32_t s = 0; s < num_strips; ++s) {
        if constexpr (NUMERIC_STABLE) {
            // ----- Pass 1: stream the full reduce dim, compute MAX -----
            // WaitAndPopPerTile keeps cb_input_tiles bounded at 2 pages — the
            // reduce holds DST across all reduce_dim_tiles tiles and packs once.
            // partial_scaler is `none()` for tile-aligned shapes and
            // `last_tile_at(1)` for Refinement 4 non-aligned shapes (the
            // overrides routed through accordingly).
            ckl::reduce<
                ckernel::PoolType::MAX,
                REDUCE_AXIS,
                ckl::ReduceInputPolicy::WaitAndPopPerTile,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
                cb_input_tiles,
                cb_max_scaler,
                cb_max,
                pass1_reduce_shape,
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::NoAccumulation{},
                ckl::NoOp{},
                partial_scaler);

            // ----- Pass 2: stream x again, per-block sub+exp → SUM, recip on last -----
            for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                // sub<BCAST>(x_block, cb_max) with exp fused into the dst-sync window.
                // cb_input_tiles is consumed per-tile (reader streaming); cb_max
                // is held across the whole strip (WaitUpfrontNoPop).
                ckl::sub<
                    BCAST_AXIS,
                    ckl::BinaryInputPolicy::WaitAndPopPerTile,
                    ckl::BinaryInputPolicy::WaitUpfrontNoPop,
                    ckl::BinaryOutputPolicy::PerTile,
                    ckl::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
                    cb_input_tiles, cb_max, cb_centered_exp, block_bin_shape, [](uint32_t dst_idx) {
                        exp_tile_init();
                        exp_tile(dst_idx);
                    });

                // accumulate_reduce_block routes the recip postop ONLY to the
                // last block (b == NUM_BLOCKS-1), so cb_inv_sum ends the loop
                // holding `1 / Σ exp(x - max)`.
                //
                // Refinement 4: the helper forwards `partial_scaler` only on
                // the last block, where the inner reduce<> uses it for the
                // last reduce-dim iteration (which is also the strip's last
                // tile, since NUM_BLOCKS * BLOCK_SIZE = reduce_dim_tiles).
                ckl::accumulate_reduce_block<ckernel::PoolType::SUM, REDUCE_AXIS>(
                    cb_centered_exp,
                    cb_sum_scaler,
                    cb_inv_sum,
                    block_reduce_shape,
                    b,
                    NUM_BLOCKS,
                    partial_scaler,
                    [](uint32_t dst_idx) {
                        // legacy_compat=false picks the Newton-Raphson recip
                        // (≤1 ulp with fp32 DEST); the default legacy path
                        // empirically lost ~10-11 bits of precision here.
                        recip_tile_init</*legacy_compat=*/false>();
                        recip_tile</*legacy_compat=*/false>(dst_idx);
                    });
            }

            // ----- Pass 3: stream x once more, per-block sub+exp → mul by inv_sum → output -----
            for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                ckl::sub<
                    BCAST_AXIS,
                    ckl::BinaryInputPolicy::WaitAndPopPerTile,
                    ckl::BinaryInputPolicy::WaitUpfrontNoPop,
                    ckl::BinaryOutputPolicy::PerTile,
                    ckl::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
                    cb_input_tiles, cb_max, cb_centered_exp, block_bin_shape, [](uint32_t dst_idx) {
                        exp_tile_init();
                        exp_tile(dst_idx);
                    });

                ckl::mul<
                    BCAST_AXIS,
                    ckl::BinaryInputPolicy::WaitAndPopPerTile,
                    ckl::BinaryInputPolicy::WaitUpfrontNoPop,
                    ckl::BinaryOutputPolicy::PerTile,
                    ckl::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
                    cb_centered_exp, cb_inv_sum, cb_output_tiles, block_bin_shape);
            }

            // The persistent-across-strip CBs each hold exactly 1 tile after
            // their final consumer (Pass 3 / mul). Drain so the next strip
            // starts with empty cb_max / cb_inv_sum.
            cb_pop_front(cb_max, 1);
            cb_pop_front(cb_inv_sum, 1);
        } else {
            // ----- Numeric_stable = False (2 passes) -----
            // Pass 1: per-block sfpu_exp → SUM, recip on last.
            for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                ckl::sfpu_exp<cb_input_tiles>(cb_centered_exp, BLOCK_SIZE);

                ckl::accumulate_reduce_block<ckernel::PoolType::SUM, REDUCE_AXIS>(
                    cb_centered_exp,
                    cb_sum_scaler,
                    cb_inv_sum,
                    block_reduce_shape,
                    b,
                    NUM_BLOCKS,
                    partial_scaler,
                    [](uint32_t dst_idx) {
                        recip_tile_init</*legacy_compat=*/false>();
                        recip_tile</*legacy_compat=*/false>(dst_idx);
                    });
            }

            // Pass 2: per-block sfpu_exp → mul by inv_sum → output.
            for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                ckl::sfpu_exp<cb_input_tiles>(cb_centered_exp, BLOCK_SIZE);

                ckl::mul<
                    BCAST_AXIS,
                    ckl::BinaryInputPolicy::WaitAndPopPerTile,
                    ckl::BinaryInputPolicy::WaitUpfrontNoPop,
                    ckl::BinaryOutputPolicy::PerTile,
                    ckl::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
                    cb_centered_exp, cb_inv_sum, cb_output_tiles, block_bin_shape);
            }

            cb_pop_front(cb_inv_sum, 1);
        }
    }
}
