// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/addcdiv.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {
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

    static ALWI void init() { copy_tile_init(Cb); }
    ALWI void wait_per_tile(uint32_t /*i*/) const {}
    ALWI void wait_upfront(uint32_t /*n*/) const {}
    ALWI void exec(uint32_t /*i*/, uint32_t /*slot_offset*/) const {
        copy_tile(Cb, 0, compute_kernel_lib::to_u32(DstSlot));
    }
    ALWI void pop_per_tile(uint32_t /*i*/) const {}
    ALWI void pop_upfront_end(uint32_t /*n*/) const {}
};

// Inherit FillTileTag — chain dispatcher routes Fill/Rand through `member exec(i)`,
// which lets us carry runtime state (scalar_arg).
struct LocalTernarySfpuStage : compute_kernel_lib::FillTileTag {
    static constexpr bool is_upfront = false;
    static constexpr bool clashes_with_fpu = false;
    static constexpr uint32_t block_size = 1;
    uint32_t scalar_arg = 0;
    static ALWI void init() { TERNARY_SFPU_OP_INIT(); }
    ALWI void exec(uint32_t /*i*/, uint32_t /*slot_offset*/) const { TERNARY_SFPU_OP_FUNC(0, 1, 2, 0, scalar_arg); }
};
}  // namespace

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c
    constexpr auto cb_out = tt::CBIndex::c_3;  // output

    unary_op_init_common(cb_in0, cb_out);

    using LoadA = LocalLoadTile<cb_in0, Dst::D0>;
    using LoadB = LocalLoadTile<cb_in1, Dst::D1>;
    using LoadC = LocalLoadTile<cb_in2, Dst::D2>;
    using PackOut = PackTile<
        cb_out,
        Dst::D0,
        PackTilePolicy::NoReserveNoPush,
        PackTileIndexMode::FirstTile,
        PackTileReconfig::None>;
    LocalTernarySfpuStage tern_stage{};
    tern_stage.scalar_arg = scalar_arg;

    auto process_tile = [&](uint32_t freq, uint32_t start) {
#if BCAST_A
        cb_wait_front(cb_in0, num_tiles_per_cycle);
#endif
#if BCAST_B
        cb_wait_front(cb_in1, num_tiles_per_cycle);
#endif
#if BCAST_C
        cb_wait_front(cb_in2, num_tiles_per_cycle);
#endif

        for (uint32_t j = start; j < freq; ++j) {
#if !BCAST_A
            cb_wait_front(cb_in0, num_tiles_per_cycle);
#endif
#if !BCAST_B
            cb_wait_front(cb_in1, num_tiles_per_cycle);
#endif
#if !BCAST_C
            cb_wait_front(cb_in2, num_tiles_per_cycle);
#endif

            cb_reserve_back(cb_out, num_tiles_per_cycle);

            // Migrated chain: load A/B/C → ternary SFPU → pack.
            eltwise_chain(1u, LoadA{}, LoadB{}, LoadC{}, tern_stage, PackOut{});

            cb_push_back(cb_out, num_tiles_per_cycle);

#if !BCAST_A
            cb_pop_front(cb_in0, num_tiles_per_cycle);
#endif
#if !BCAST_B
            cb_pop_front(cb_in1, num_tiles_per_cycle);
#endif
#if !BCAST_C
            cb_pop_front(cb_in2, num_tiles_per_cycle);
#endif
        }

#if BCAST_A
        cb_pop_front(cb_in0, num_tiles_per_cycle);
#endif
#if BCAST_B
        cb_pop_front(cb_in1, num_tiles_per_cycle);
#endif
#if BCAST_C
        cb_pop_front(cb_in2, num_tiles_per_cycle);
#endif
    };

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(tile_freq, tile_start);
    }
    if (remaining_iterations > 0) {
        process_tile(remaining_iterations, tile_start);
    }
}
