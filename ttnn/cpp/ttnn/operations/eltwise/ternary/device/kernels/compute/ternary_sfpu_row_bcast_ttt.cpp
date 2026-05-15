// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Ternary SFPU compute kernel with optional ROW broadcast via LLK unary_bcast
// Supports broadcasting 0, 1 or 2 inputs (A/B/C)
// Expects reader kernel to not perform per-tile fill (FILL_TILE_WITH_FIRST_ROW) controlled by BCAST_LLK flag
//
// The actual ternary operation is provided via:
//   TERNARY_SFPU_OP_INIT()
//   TERNARY_SFPU_OP_FUNC(src_idx_a, src_idx_b, src_idx_c, dst_idx_out)
// which are configured by the program factory for the desired op (e.g., where).

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
// Local chain elements for the ternary SFPU stage:
//   D[0] = A, D[1] = B, D[2] = C, then TERNARY_SFPU_OP_FUNC(0,1,2,0).
// Outer scope owns input wait/pop; chain elements use NoWaitNoPop for inputs/outputs.
template <uint32_t Cb, compute_kernel_lib::Dst DstSlot>
struct LocalLoadTile : compute_kernel_lib::CopyTileTag {
    static constexpr uint32_t cb = Cb;
    static constexpr uint32_t cb_a_id() { return Cb; }
    static constexpr uint32_t cb_b_id() { return 0; }
    static constexpr compute_kernel_lib::Dst dst_slot = DstSlot;
    static constexpr compute_kernel_lib::CopyTilePolicy a_policy() {
        return compute_kernel_lib::CopyTilePolicy::NoWaitNoPop;
    }
    static constexpr compute_kernel_lib::CopyTilePolicy b_policy() {
        return compute_kernel_lib::CopyTilePolicy::NoWaitNoPop;
    }
    static constexpr bool is_upfront = false;
    static constexpr bool clashes_with_fpu = true;
    static constexpr uint32_t block_size = 1;

    static ALWI void init() { copy_tile_to_dst_init_short(Cb); }
    ALWI void wait_per_tile(uint32_t /*i*/) const {}
    ALWI void wait_upfront(uint32_t /*n*/) const {}
    ALWI void exec(uint32_t /*i*/, uint32_t /*slot_offset*/) const {
        copy_tile(Cb, 0, compute_kernel_lib::to_u32(DstSlot));
    }
    ALWI void pop_per_tile(uint32_t /*i*/) const {}
    ALWI void pop_upfront_end(uint32_t /*n*/) const {}
};

struct LocalTernarySfpuStage : compute_kernel_lib::DestOnlyTag {
    static constexpr bool is_upfront = false;
    static constexpr bool clashes_with_fpu = false;
    static constexpr uint32_t block_size = 1;
    static ALWI void init() { TERNARY_SFPU_OP_INIT(); }
    ALWI void exec(uint32_t /*i*/, uint32_t /*slot_offset*/) const { TERNARY_SFPU_OP_FUNC(0, 1, 2, 0); }
};
}  // namespace

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // typically 1

    // Pre-CBs for inputs A, B, C and output CB
    constexpr auto cb_pre_a = tt::CBIndex::c_0;
    constexpr auto cb_pre_b = tt::CBIndex::c_1;
    constexpr auto cb_pre_c = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    // CBs to hold LLK broadcast results when enabled
    constexpr auto cb_bcast_a = tt::CBIndex::c_4;
    constexpr auto cb_bcast_b = tt::CBIndex::c_5;
    constexpr auto cb_bcast_c = tt::CBIndex::c_6;

// Compile-time effective CB selection
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

    // Initialize pack/format for SFPU-style ternary kernels (matches existing ternary SFPU kernels)
    unary_op_init_common(cb_eff_a, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
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

        // Now execute the ternary SFPU operation on the effective inputs.
        // Reserve output when ready to write.
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        // Ensure fronts are ready for whichever CBs we will read from
        cb_wait_front(cb_eff_a, num_tiles_per_cycle);
        cb_wait_front(cb_eff_b, num_tiles_per_cycle);
        cb_wait_front(cb_eff_c, num_tiles_per_cycle);

        // Migrated stage: ternary SFPU loads + TERNARY_SFPU_OP via chain.
        using LoadA = LocalLoadTile<cb_eff_a, compute_kernel_lib::Dst::D0>;
        using LoadB = LocalLoadTile<cb_eff_b, compute_kernel_lib::Dst::D1>;
        using LoadC = LocalLoadTile<cb_eff_c, compute_kernel_lib::Dst::D2>;
        using PackOut = compute_kernel_lib::PackTile<
            cb_out,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::PackTilePolicy::NoReserveNoPush,
            compute_kernel_lib::PackTileIndexMode::FirstTile,
            compute_kernel_lib::PackTileReconfig::None>;
        compute_kernel_lib::eltwise_chain(1u, LoadA{}, LoadB{}, LoadC{}, LocalTernarySfpuStage{}, PackOut{});

        cb_push_back(cb_out, num_tiles_per_cycle);

        // Pop fronts for the consumed inputs.
        cb_pop_front(cb_eff_a, num_tiles_per_cycle);
        cb_pop_front(cb_eff_b, num_tiles_per_cycle);
        cb_pop_front(cb_eff_c, num_tiles_per_cycle);
    }
}
