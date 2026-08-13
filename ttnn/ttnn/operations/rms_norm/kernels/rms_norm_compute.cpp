// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm compute.  Realizes op_design.md's named block operations:
//
//   tilize_block            -> compute_kernel_lib::tilize            (ROW_MAJOR only)
//   mask_tail_block         -> eltwise_chain (BinaryFpu Mul, bcast Row, strided, in place)
//   square_accumulate_block -> compute_kernel_lib::sum_of_squares
//   collapse_partial_block  -> compute_kernel_lib::reduce<SUM, REDUCE_ROW, ..., Collapse>
//   combine_block (root)    -> compute_kernel_lib::reduce<SUM, REDUCE_ROW, ..., Skip>
//   scale_block             -> compute_kernel_lib::mul (bcast Col)
//   apply_gamma_block       -> compute_kernel_lib::mul (bcast Row)
//   untilize_block          -> compute_kernel_lib::untilize          (ROW_MAJOR only)
//
// Raw-LLK notes (deviations from "prefer helpers"):
//  * The finalize (x1/W, +epsilon, rsqrt) is written as raw SFPU calls inside a
//    `post_reduce_op` lambda.  That lambda IS the reduce helper's documented
//    extension point (reduce_helpers_compute.hpp:491-495): running the three ops
//    on the reduce's DEST tile is what keeps `mean+eps` from ever being packed
//    to L1.  Expressing it as a separate eltwise_chain would need a whole extra
//    B-page CB and an L1 round trip.
//  * `mask_tail_block` uses `eltwise_chain` rather than the `mul` convenience
//    wrapper because the convenience wrappers default-construct their elements
//    and so cannot carry a `StridedTileRange` (the strided, in-place window over
//    one tile per tile-row).  Same helper, same chain, explicit element ctors.
//
// x lives in cb_input_tiles across THREE phases and is rewritten in place twice.
// The compute kernel owns exactly one cb_wait_front / cb_pop_front window per
// block; every chain that touches it uses WaitPolicy::None / PopPolicy::None /
// ReservePolicy::None / PushPolicy::None so no helper issues a competing
// handshake.  cb_input_tiles' capacity is exactly BLOCK_ROWS*S, which is what
// makes get_write_ptr() == get_read_ptr() and the in-place pack correct.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gamma_tiles = 1;
constexpr uint32_t cb_sq_partials = 2;
constexpr uint32_t cb_slice_stat = 3;
constexpr uint32_t cb_gathered_partials = 4;
constexpr uint32_t cb_rms_bcast = 5;
constexpr uint32_t cb_rms_recip = 6;
constexpr uint32_t cb_scaler = 7;
constexpr uint32_t cb_w_mask = 8;
constexpr uint32_t cb_output_tiles = 9;
constexpr uint32_t cb_rm_stage_in = 10;
constexpr uint32_t cb_rm_stage_out = 11;

constexpr uint32_t NO_MASK_COL = 0xFFFFFFFFu;

void kernel_main() {
    // ---- block knobs ----
    constexpr uint32_t SLICE_HIDDEN_TILES = get_compile_time_arg_val(0);  // S
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(1);          // B
    constexpr uint32_t NUM_HIDDEN_SLICES = get_compile_time_arg_val(2);   // s
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(3);
    constexpr uint32_t IS_ROW_MAJOR = get_compile_time_arg_val(4);
    constexpr uint32_t MASK_ENABLED = get_compile_time_arg_val(5);

    constexpr uint32_t BLOCK_TILES = BLOCK_ROWS * SLICE_HIDDEN_TILES;

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);
    const uint32_t mask_local_col = get_arg_val<uint32_t>(2);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(3);
    const uint32_t eps_bits = get_arg_val<uint32_t>(4);

    if constexpr (IS_ROW_MAJOR) {
        compute_kernel_hw_startup(cb_rm_stage_in, cb_scaler, cb_input_tiles);
    } else {
        compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles);
    }

    // Resident constants: waited once, never popped.
    if constexpr (HAS_GAMMA) {
        cb_wait_front(cb_gamma_tiles, SLICE_HIDDEN_TILES);
    }
    const bool do_mask = (MASK_ENABLED != 0) && (mask_local_col != NO_MASK_COL);
    if (do_mask) {
        cb_wait_front(cb_w_mask, 1);
    }

    // finalize: mean = Sum(x^2) * (1/W) using the TRUE element count W, then
    // + epsilon, then rsqrt — applied exactly ONCE, after the cross-core combine.
    // ---- operand configurations (compile-time; every chain that touches
    //      cb_input_tiles is caller-managed so no helper competes for its
    //      wait/pop/reserve/push window) ----
    constexpr auto x_held =
        ckl::input(cb_input_tiles, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block);
    constexpr auto rms_col = ckl::input(
        cb_rms_recip,
        ckl::BroadcastDim::Col,
        ckl::WaitPolicy::Upfront,
        ckl::PopPolicy::AtEnd,
        ckl::OperandKind::Col,
        ckl::TileOffset::Unset);
    constexpr auto gamma_row = ckl::input(
        cb_gamma_tiles,
        ckl::BroadcastDim::Row,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::OperandKind::Row,
        ckl::TileOffset::Unset);
    constexpr auto in_place = ckl::output(cb_input_tiles, ckl::ReservePolicy::None, ckl::PushPolicy::None);
    // Per-tile reserve/push: the chain reserves (PerOuter, PerOuter) exclusively
    // for DEST-accumulating packs, so the streaming output CB uses the per-tile
    // lifecycle.  The writer still drains a whole tile-row (S pages) at a time,
    // so cb_output_tiles' out_cb_depth window is what buys the overlap.
    constexpr auto to_output = ckl::output(cb_output_tiles);
    constexpr auto block_shape = ckl::IterationShape::grid(BLOCK_ROWS, SLICE_HIDDEN_TILES);

    auto finalize = [inv_w_bits, eps_bits](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_idx, inv_w_bits);
        add_unary_tile(dst_idx, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    };

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // ---- tilize_block (ROW_MAJOR only) ----
        if constexpr (IS_ROW_MAJOR) {
            ckl::tilize<SLICE_HIDDEN_TILES, cb_rm_stage_in, cb_input_tiles>(BLOCK_ROWS);
        }

        // ---- the single cb_input_tiles window for this block ----
        cb_wait_front(cb_input_tiles, BLOCK_TILES);

        // ---- mask_tail_block: zero the W-pad lanes of the LAST hidden tile of
        //      each tile-row, in place. Only on the core owning the global last
        //      hidden tile, and only when W % 32 != 0 under TILE layout.
        if constexpr (MASK_ENABLED) {
            if (do_mask) {
                const ckl::StridedTileRange window{mask_local_col, SLICE_HIDDEN_TILES};
                ckl::eltwise_chain(
                    ckl::IterationShape::grid(BLOCK_ROWS, 1),
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Mul,
                        ckl::input(
                            cb_input_tiles,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Col,
                            ckl::TileOffset::Strided),
                        ckl::input(cb_w_mask, ckl::BroadcastDim::Row, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{
                        window},
                    ckl::PackTile<ckl::output(
                        cb_input_tiles, ckl::ReservePolicy::None, ckl::PushPolicy::None, ckl::TileOffset::Strided)>{
                        window});
            }
        }

        // ---- square_accumulate_block: Sum over the slice of x*x, folded in DEST
        //      per tile-row (no x^2 tiles are ever materialized). x is HELD.
        ckl::sum_of_squares<x_held, ckl::row_output(cb_sq_partials)>(block_shape);

        if constexpr (NUM_HIDDEN_SLICES > 1) {
            // ---- collapse_partial_block: within-tile collapse -> column-0-valid
            //      per-slice partial. NO finalize here: 1/W and epsilon must be
            //      applied once, after the combine.
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_sq_partials,
                cb_scaler,
                cb_slice_stat,
                ckl::ReduceInputPolicy::BulkWaitBulkPop>(ckl::ReduceInputBlockShape::of(BLOCK_ROWS, 1));

            // ---- combine_block (root): sum the s gathered column-0-valid
            //      partials. ReduceWithinTile::Skip is the documented case —
            //      the inputs are already collapsed on the reduce axis.
            if (is_root) {
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_gathered_partials,
                    cb_scaler,
                    cb_rms_bcast,
                    ckl::ReduceInputPolicy::BulkWaitBulkPop,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ReduceFp32Mode::Fast,
                    ckl::ReduceAlgorithm::AccumulateViaAdd,
                    ckl::NoAccumulation,
                    decltype(finalize),
                    ckl::ReduceWithinTile::Skip>(
                    ckl::ReduceInputBlockShape::of(BLOCK_ROWS, NUM_HIDDEN_SLICES),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::NoAccumulation{},
                    finalize);
            }
        } else {
            // ---- collapse_partial_block with the finalize fused in ----
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_sq_partials,
                cb_scaler,
                cb_rms_recip,
                ckl::ReduceInputPolicy::BulkWaitBulkPop,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                ckl::ReduceAlgorithm::Auto,
                ckl::NoAccumulation,
                decltype(finalize)>(
                ckl::ReduceInputBlockShape::of(BLOCK_ROWS, 1),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::NoAccumulation{},
                finalize);
        }

        // ---- scale_block: x *= rsqrt(mean + eps). A REDUCE_ROW result is
        //      column-shaped, so it broadcasts back across columns (Col).
        if constexpr (HAS_GAMMA || IS_ROW_MAJOR) {
            ckl::mul<x_held, rms_col, in_place>(block_shape);
        } else {
            ckl::mul<x_held, rms_col, to_output>(block_shape);
        }

        // ---- apply_gamma_block: gamma is a 1D [W] operand -> Row broadcast ----
        if constexpr (HAS_GAMMA) {
            if constexpr (IS_ROW_MAJOR) {
                ckl::mul<x_held, gamma_row, in_place>(block_shape);
            } else {
                ckl::mul<x_held, gamma_row, to_output>(block_shape);
            }
        }

        // ---- untilize_block (ROW_MAJOR) / release the window (TILE) ----
        if constexpr (IS_ROW_MAJOR) {
            // NoWait: compute already holds the BLOCK_TILES window; untilize's
            // per-tile-row pop IS the window's release.
            ckl::untilize<
                SLICE_HIDDEN_TILES,
                cb_input_tiles,
                cb_rm_stage_out,
                ckl::untilize_config::InitUninitMode::InitAndUninit,
                ckl::untilize_config::WaitMode::NoWait>(BLOCK_ROWS);
        } else {
            cb_pop_front(cb_input_tiles, BLOCK_TILES);
        }
    }
}
