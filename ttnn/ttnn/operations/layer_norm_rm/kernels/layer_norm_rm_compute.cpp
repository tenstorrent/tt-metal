// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// layer_norm_rm compute kernel — per op_design.md Phase 0.
//
// Per work-item (one tile-row):
//   Phase 1: tilize cb_input_rm → cb_input_tiles
//   Phase 2: reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop> over cb_input_tiles
//            with scaler 1/W → cb_mean (column-vector tile, col-0 valid)
//   Phase 3: sub<COL, WaitUpfrontPopAtEnd, WaitUpfrontPopAtEnd> of
//            cb_input_tiles − cb_mean → cb_centered
//   Phase 4: square<WaitUpfrontNoPop> of cb_centered → cb_centered_sq
//            (leaves cb_centered for Phase 6's mul)
//   Phase 5: reduce<SUM, REDUCE_ROW> over cb_centered_sq with scaler 1/W and
//            post-op (+ eps, then rsqrt) → cb_inv_std
//            The post-op runs inside the reduce's dst-sync window before
//            pack — one LLK fusion produces 1/sqrt(var + eps).
//   Phase 6: mul<COL, WaitUpfrontPopAtEnd, WaitUpfrontPopAtEnd> of
//            cb_centered * cb_inv_std → cb_norm  (normalized output)
//   Phase 7 (has_gamma): mul_in_place<ROW, NoWaitNoPop>(cb_norm, cb_gamma_tiles)
//   Phase 8 (has_beta):  add_in_place<ROW, NoWaitNoPop>(cb_norm, cb_beta_tiles)
//   Phase 9: untilize<Wt>(cb_norm → cb_output_tiles)
//
// Boot (executed once, before the loop):
//   - compute_kernel_hw_startup(cb_input_rm, cb_scaler, cb_output_tiles).
//     Three-arg form is the safe default across the chained helpers (advisory
//     deviation from the design's two-arg `(cb_input_rm, cb_input_tiles)`).
//   - If has_gamma: tilize gamma RM → cb_gamma_tiles, then one-shot
//     cb_wait_front(cb_gamma_tiles, Wt) so subsequent NoWaitNoPop uses are
//     legal.
//   - If has_beta: same for beta.
//
// `fp32_dest_acc_en=true` halves DEST half-sync capacity to 4 tiles. All
// helpers respect DEST_AUTO_LIMIT internally. The Phase-5 post-op operates
// on a single DST register (the reduce's accumulator) — safe.
//
// cb_norm must be exclusively compute-owned (binary_op_helpers.hpp:340-348):
// no reader/writer push or pop. Verified here — reader fills cb_input_rm /
// cb_gamma_rm / cb_beta_rm / cb_scaler only; writer drains cb_output_tiles.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"

#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace {
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_gamma_rm = 1;
constexpr uint32_t cb_beta_rm = 2;
constexpr uint32_t cb_scaler = 8;
constexpr uint32_t cb_gamma_tiles = 9;
constexpr uint32_t cb_beta_tiles = 10;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_input_tiles = 24;
constexpr uint32_t cb_mean = 25;
constexpr uint32_t cb_centered = 26;
constexpr uint32_t cb_centered_sq = 27;
constexpr uint32_t cb_inv_std = 28;
constexpr uint32_t cb_norm = 29;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t has_beta = get_compile_time_arg_val(2);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(3);  // fp32 bits of epsilon

    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    // ------- One-shot hardware startup (must precede every helper) -------
    compute_kernel_hw_startup(cb_input_rm, cb_scaler, cb_output_tiles);

    // ------- Boot: tilize gamma / beta into their persistent tile CBs -------
    if constexpr (has_gamma) {
        ckl::tilize<Wt, cb_gamma_rm, cb_gamma_tiles>(1);
        // Make subsequent NoWaitNoPop uses (binary_op_helpers.hpp:141-146) legal.
        cb_wait_front(cb_gamma_tiles, Wt);
    }
    if constexpr (has_beta) {
        ckl::tilize<Wt, cb_beta_rm, cb_beta_tiles>(1);
        cb_wait_front(cb_beta_tiles, Wt);
    }

    // ------- Shape constants reused inside the loop -------
    constexpr auto reduce_shape = ckl::ReduceInputBlockShape::of(1, Wt);
    constexpr auto bin_shape = ckl::BinaryInputBlockShape::of(1, Wt);

    // ------- Per-work-item pipeline -------
    for (uint32_t i = 0; i < num_tile_rows; ++i) {
        // Phase 1: tilize 32 sticks → Wt tiles.
        ckl::tilize<Wt, cb_input_rm, cb_input_tiles>(1);

        // Phase 2: MEAN reduce (scaler = 1/W → SUM produces mean directly).
        //   WaitUpfrontNoPop leaves cb_input_tiles intact for Phase 3.
        ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
            cb_input_tiles, cb_scaler, cb_mean, reduce_shape);

        // Phase 3: SUB — broadcast mean (col-vector tile) across the Wt tiles.
        //   WaitUpfrontPopAtEnd on A pairs with Phase 2's WaitUpfrontNoPop on
        //   the same CB; data already present, pop at end.
        ckl::sub<
            ckl::BroadcastDim::COL,
            ckl::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            ckl::BinaryInputPolicy::WaitUpfrontPopAtEnd>(cb_input_tiles, cb_mean, cb_centered, bin_shape);

        // Phase 4: SQUARE (out-of-place) — leaves cb_centered for Phase 6.
        ckl::square<ckl::BinaryInputPolicy::WaitUpfrontNoPop>(cb_centered, cb_centered_sq, bin_shape);

        // Phase 5: VARIANCE reduce + (+eps, rsqrt) post-op → inv_std.
        //   Post-op runs inside the reduce's dst-sync window before pack.
        //   add_unary_tile uses the binop_with_scalar SFPU init.
        ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, ckl::ReduceInputPolicy::WaitAndPopPerTile>(
            cb_centered_sq,
            cb_scaler,
            cb_inv_std,
            reduce_shape,
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t dst_idx) {
                // var + epsilon, then rsqrt — fused in DST.
                ckernel::binop_with_scalar_tile_init();
                ckernel::add_unary_tile(dst_idx, eps_bits);
                ckernel::rsqrt_tile_init</*legacy_compat=*/false>();
                ckernel::rsqrt_tile</*legacy_compat=*/false>(dst_idx);
            });

        // Phase 6: MUL by inv_std (col-broadcast) → normalized output.
        ckl::mul<
            ckl::BroadcastDim::COL,
            ckl::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            ckl::BinaryInputPolicy::WaitUpfrontPopAtEnd>(cb_centered, cb_inv_std, cb_norm, bin_shape);

        // Phase 7: optional in-place mul by gamma (row-broadcast).
        //   Gamma was pre-waited at boot — NoWaitNoPop is safe.
        if constexpr (has_gamma) {
            ckl::mul_in_place<ckl::BroadcastDim::ROW, ckl::BinaryInputPolicy::NoWaitNoPop>(
                cb_norm, cb_gamma_tiles, bin_shape);
        }

        // Phase 8: optional in-place add of beta (row-broadcast).
        if constexpr (has_beta) {
            ckl::add_in_place<ckl::BroadcastDim::ROW, ckl::BinaryInputPolicy::NoWaitNoPop>(
                cb_norm, cb_beta_tiles, bin_shape);
        }

        // Phase 9: untilize → cb_output_tiles for the writer.
        ckl::untilize<Wt, cb_norm, cb_output_tiles>(1);
    }
}
