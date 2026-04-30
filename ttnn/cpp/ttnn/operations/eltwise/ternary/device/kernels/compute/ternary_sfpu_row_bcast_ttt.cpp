// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Ternary SFPU compute kernel with optional ROW broadcast via LLK unary_bcast.
// PARTIAL MIGRATION: the broadcast pre-processing stages (unary_bcast over
// cb_pre_X → cb_bcast_X, conditional on BCAST_X) remain on raw LLK because
// they are a separate compute phase with their own DEST acquire/release
// cycle. The post-broadcast ternary block (copy×3 + TERNARY_SFPU_OP +
// pack) is migrated to V2 helper via TernaryMacroOp + eltwise_pipeline.

#include <cstdint>

#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {
struct TernaryMacroOp : compute_kernel_lib::TernaryOp<
                            TernaryMacroOp,
                            compute_kernel_lib::Dst::D0,
                            compute_kernel_lib::Dst::D1,
                            compute_kernel_lib::Dst::D2,
                            compute_kernel_lib::Dst::D0> {
    static constexpr bool clobbers_sfpu_lut = true;

    ALWI static void init() { TERNARY_SFPU_OP_INIT(); }
    ALWI static void call(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t out_idx) {
        TERNARY_SFPU_OP_FUNC(i0, i1, i2, out_idx);
    }
};
}  // namespace

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // typically 1
    static_assert(num_tiles_per_cycle == 1, "row-bcast TTT path runs one tile per chain invocation");

    constexpr auto cb_pre_a = tt::CBIndex::c_0;
    constexpr auto cb_pre_b = tt::CBIndex::c_1;
    constexpr auto cb_pre_c = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    constexpr auto cb_bcast_a = tt::CBIndex::c_4;
    constexpr auto cb_bcast_b = tt::CBIndex::c_5;
    constexpr auto cb_bcast_c = tt::CBIndex::c_6;

#if BCAST_A
    constexpr auto cb_eff_a = cb_bcast_a;
#else
    constexpr auto cb_eff_a = cb_pre_a;
#endif
#if BCAST_B
    constexpr auto cb_eff_b = cb_bcast_b;
#else
    constexpr auto cb_eff_b = cb_pre_b;
#endif
#if BCAST_C
    constexpr auto cb_eff_c = cb_bcast_c;
#else
    constexpr auto cb_eff_c = cb_pre_c;
#endif

    unary_op_init_common(cb_eff_a, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // ---- Broadcast pre-processing (raw LLK; partial-migration boundary) ----
#if BCAST_A
        {
            cb_wait_front(cb_pre_a, num_tiles_per_cycle);
            cb_reserve_back(cb_bcast_a, num_tiles_per_cycle);
            unary_bcast_init<BroadcastType::ROW>(cb_pre_a, cb_bcast_a);
            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(cb_pre_a, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_bcast_a);
            cb_push_back(cb_bcast_a, num_tiles_per_cycle);
            tile_regs_release();
            cb_pop_front(cb_pre_a, num_tiles_per_cycle);
        }
#endif
#if BCAST_B
        {
            cb_wait_front(cb_pre_b, num_tiles_per_cycle);
            cb_reserve_back(cb_bcast_b, num_tiles_per_cycle);
            unary_bcast_init<BroadcastType::ROW>(cb_pre_b, cb_bcast_b);
            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(cb_pre_b, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_bcast_b);
            cb_push_back(cb_bcast_b, num_tiles_per_cycle);
            tile_regs_release();
            cb_pop_front(cb_pre_b, num_tiles_per_cycle);
        }
#endif
#if BCAST_C
        {
            cb_wait_front(cb_pre_c, num_tiles_per_cycle);
            cb_reserve_back(cb_bcast_c, num_tiles_per_cycle);
            unary_bcast_init<BroadcastType::ROW>(cb_pre_c, cb_bcast_c);
            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(cb_pre_c, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_bcast_c);
            cb_push_back(cb_bcast_c, num_tiles_per_cycle);
            tile_regs_release();
            cb_pop_front(cb_pre_c, num_tiles_per_cycle);
        }
#endif

        // ---- Ternary SFPU op (V2 helper) ----
        compute_kernel_lib::eltwise_pipeline<cb_out>(
            num_tiles_per_cycle,
            compute_kernel_lib::eltwise_chain(
                compute_kernel_lib::CopyTile<cb_eff_a, compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::CopyTile<cb_eff_b, compute_kernel_lib::Dst::D1>{},
                compute_kernel_lib::CopyTile<cb_eff_c, compute_kernel_lib::Dst::D2>{},
                TernaryMacroOp{}));
    }
}
