// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Softmax compute kernel.
//
// Reduces a strip of fp32 tiles (1×Wt for dim=-1, Ht×1 for dim=-2) to a
// numerically-stable softmax strip, using the helper library:
//
//   numeric_stable = True (default, 4 phases):
//     Phase A: reduce<MAX, ReduceDim, WaitUpfrontNoPop>(cb_input_tiles,
//              cb_max_scaler, cb_max)                         — keeps input resident
//     Phase B: sub<BroadcastDim, WaitUpfrontPopAtEnd, WaitUpfrontPopAtEnd>(
//              cb_input_tiles, cb_max, cb_exps,
//              postop = exp_tile)                            — drains both inputs
//     Phase C: reduce<SUM, ReduceDim, WaitUpfrontNoPop>(cb_exps,
//              cb_sum_scaler, cb_inv_sum,
//              postop = recip_tile)                          — keeps exps resident
//     Phase D: mul<BroadcastDim, WaitUpfrontPopAtEnd, WaitUpfrontPopAtEnd>(
//              cb_exps, cb_inv_sum, cb_output_tiles)         — drains both inputs
//
//   numeric_stable = False (3 phases — Phase B′ replaces A+B):
//     Phase B′: sfpu_exp<cb_input_tiles>(cb_exps, reduce_dim_tiles)
//     Phase C, D: as above.
//
// Per (dim, helper) template choices:
//     dim = -1 (REDUCE_ROW):
//         ReduceDim    = REDUCE_ROW
//         BroadcastDim = COL          (column-vector output from REDUCE_ROW,
//                                      broadcast across the columns of each tile)
//         block shape  = (1, Wt)
//     dim = -2 (REDUCE_COL):
//         ReduceDim    = REDUCE_COL
//         BroadcastDim = ROW
//         block shape  = (Ht, 1)
//
// CB sync (per strip):
//   cb_input_tiles : reader pushes reduce_dim_tiles  → Phase A waits upfront
//                    (NoPop)  → Phase B waits upfront (PopAtEnd) drains it.
//   cb_max         : Phase A pushes 1 → Phase B PopAtEnd drains.
//   cb_exps        : Phase B pushes reduce_dim_tiles → Phase C waits upfront
//                    (NoPop) → Phase D PopAtEnd drains.
//   cb_inv_sum     : Phase C pushes 1 → Phase D PopAtEnd drains.
//   cb_output_tiles: Phase D pushes reduce_dim_tiles → writer drains.
//
// Scaler CBs are one-shot at boot (reader-side) and the reduce helpers use
// WaitUpfrontNoPop, so they remain populated for the entire kernel lifetime.

#include <cstdint>

// compute_kernel_api.h brings `using namespace ckernel;` (via chlkc_list.h),
// the LLK MATH/PACK/UNPACK macros, and the standard compute API surface.
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"

#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"

namespace {
constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_max_scaler = 8;
constexpr uint32_t cb_sum_scaler = 9;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_max = 24;
constexpr uint32_t cb_exps = 25;
constexpr uint32_t cb_inv_sum = 26;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t dim_is_row = get_compile_time_arg_val(0);
    constexpr uint32_t numeric_stable_flag = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t reduce_dim_tiles = get_compile_time_arg_val(4);
    (void)Ht;
    (void)Wt;

    const uint32_t num_strips = get_arg_val<uint32_t>(0);

    // ---- One-shot hardware startup (must come before any helper) ----
    // Pair (cb_input_tiles, cb_max_scaler) for SrcA/SrcB and cb_output_tiles
    // for the packer. Helpers reconfig as needed between phases via
    // INPUT_AND_OUTPUT reconfig modes.
    compute_kernel_hw_startup(cb_input_tiles, cb_max_scaler, cb_output_tiles);

    // ---- Dim-dependent helper shape and template choices ----
    // For dim=-1 (REDUCE_ROW): shape = (1, Wt), broadcast COL.
    // For dim=-2 (REDUCE_COL): shape = (Ht, 1), broadcast ROW.
    constexpr auto reduce_shape =
        (dim_is_row != 0) ? ckl::ReduceInputBlockShape::of(1, Wt) : ckl::ReduceInputBlockShape::of(Ht, 1);

    constexpr auto bin_shape =
        (dim_is_row != 0) ? ckl::BinaryInputBlockShape::of(1, Wt) : ckl::BinaryInputBlockShape::of(Ht, 1);

    constexpr auto REDUCE_AXIS = (dim_is_row != 0) ? ckernel::ReduceDim::REDUCE_ROW : ckernel::ReduceDim::REDUCE_COL;

    constexpr auto BCAST_AXIS = (dim_is_row != 0) ? ckl::BroadcastDim::COL : ckl::BroadcastDim::ROW;

    constexpr bool NUMERIC_STABLE = (numeric_stable_flag != 0);

    // ---- Strip loop ----
    for (uint32_t s = 0; s < num_strips; ++s) {
        if constexpr (NUMERIC_STABLE) {
            // ----- Phase A: MAX reduce -----
            // reduce<MAX, ReduceDim, WaitUpfrontNoPop>(cb_input_tiles,
            //         cb_max_scaler, cb_max, shape)
            // WaitUpfrontNoPop keeps cb_input_tiles populated for Phase B.
            ckl::reduce<
                ckernel::PoolType::MAX,
                REDUCE_AXIS,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
                cb_input_tiles, cb_max_scaler, cb_max, reduce_shape);

            // ----- Phase B: sub(input, max) → exp postop -----
            // sub<BroadcastDim, WaitUpfrontPopAtEnd, WaitUpfrontPopAtEnd>(
            //     cb_input_tiles, cb_max, cb_exps, shape, exp_postop)
            // Pops cb_input_tiles (the WaitUpfrontNoPop residue from Phase A)
            // and cb_max at the end. The postop fuses exp into the dst-sync
            // window before pack, so cb_exps holds exp(x - max).
            ckl::sub<
                BCAST_AXIS,
                ckl::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                ckl::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                ckl::BinaryOutputPolicy::PerTile,
                ckl::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
                cb_input_tiles, cb_max, cb_exps, bin_shape, [](uint32_t dst_idx) {
                    exp_tile_init();
                    exp_tile(dst_idx);
                });
        } else {
            // ----- Phase B′: sfpu_exp(input) → exps -----
            // sfpu_exp<cb_input_tiles>(cb_exps, reduce_dim_tiles)
            // Streams tiles from cb_input_tiles, applies SFPU exp, packs into
            // cb_exps. The default policy WaitAndPopPerTile drains
            // cb_input_tiles as it goes — no Phase A residue here.
            ckl::sfpu_exp<cb_input_tiles>(cb_exps, reduce_dim_tiles);
        }

        // ----- Phase C: SUM reduce → recip postop -----
        // reduce<SUM, ReduceDim, WaitUpfrontNoPop>(cb_exps, cb_sum_scaler,
        //         cb_inv_sum, shape, ..., recip_postop)
        // WaitUpfrontNoPop keeps cb_exps populated for Phase D. recip_postop
        // turns Σexp into 1/Σexp so Phase D can multiply instead of divide.
        ckl::reduce<
            ckernel::PoolType::SUM,
            REDUCE_AXIS,
            ckl::ReduceInputPolicy::WaitUpfrontNoPop,
            ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            cb_exps,
            cb_sum_scaler,
            cb_inv_sum,
            reduce_shape,
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t dst_idx) {
                // legacy_compat=false dispatches to the newer Newton-Raphson recip
                // (`_calculate_reciprocal_internal_`) which, paired with fp32 DEST,
                // claims ≤1 ulp precision (recip.h comment in ckernel_sfpu_recip.h).
                // The default legacy_compat=true path uses an older recip formulation
                // that empirically lost ~10-11 bits of precision in this kernel.
                recip_tile_init</*legacy_compat=*/false>();
                recip_tile</*legacy_compat=*/false>(dst_idx);
            });

        // ----- Phase D: mul(exps, 1/Σexp) -----
        // mul<BroadcastDim, WaitUpfrontPopAtEnd, WaitUpfrontPopAtEnd>(
        //     cb_exps, cb_inv_sum, cb_output_tiles, shape)
        // Drains cb_exps (the WaitUpfrontNoPop residue from Phase C) and
        // cb_inv_sum. Result: softmax(x) tiles in cb_output_tiles, ready
        // for the writer.
        ckl::mul<
            BCAST_AXIS,
            ckl::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            ckl::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            ckl::BinaryOutputPolicy::PerTile,
            ckl::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(cb_exps, cb_inv_sum, cb_output_tiles, bin_shape);
    }
}
