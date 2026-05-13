// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"

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

struct LocalTernarySfpuStage : compute_kernel_lib::DestOnlyTag {
    static constexpr bool is_upfront = false;
    static constexpr bool clashes_with_fpu = false;
    static constexpr uint32_t block_size = 1;
    static ALWI void init() { TERNARY_SFPU_OP_INIT(); }
    ALWI void exec(uint32_t /*i*/, uint32_t /*slot_offset*/) const { TERNARY_SFPU_OP_FUNC(0, 1, 2, 0); }
};
}  // namespace

template <tt::CBIndex predicate_cb, tt::CBIndex true_cb, tt::CBIndex false_cb, tt::CBIndex cb_out>
ALWI void process_tile(uint32_t freq, uint32_t tile_start, uint32_t num_tiles_per_cycle) {
    using namespace ckernel;
    using namespace compute_kernel_lib;

    // 3-tensor broadcast-aware synchronization - wait for broadcast CBs outside loop
#if BCAST_A
    cb_wait_front(predicate_cb, num_tiles_per_cycle);  // predicate_cb is broadcast
#endif
#if BCAST_B
    cb_wait_front(true_cb, num_tiles_per_cycle);  // true_cb is broadcast
#endif
#if BCAST_C
    cb_wait_front(false_cb, num_tiles_per_cycle);  // false_cb is broadcast
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
        // Wait for non-broadcast CBs inside loop
#if !BCAST_A
        cb_wait_front(predicate_cb, num_tiles_per_cycle);
#endif
#if !BCAST_B
        cb_wait_front(true_cb, num_tiles_per_cycle);
#endif
#if !BCAST_C
        cb_wait_front(false_cb, num_tiles_per_cycle);
#endif

        cb_reserve_back(cb_out, num_tiles_per_cycle);

        // Migrated stage: load A/B/C → ternary SFPU → pack via eltwise_chain.
        using LoadA = LocalLoadTile<(uint32_t)predicate_cb, Dst::D0>;
        using LoadB = LocalLoadTile<(uint32_t)true_cb, Dst::D1>;
        using LoadC = LocalLoadTile<(uint32_t)false_cb, Dst::D2>;
        using PackOut = PackTile<(uint32_t)cb_out, Dst::D0, PackTilePolicy::NoReserveNoPush>;
        eltwise_chain(1u, LoadA{}, LoadB{}, LoadC{}, LocalTernarySfpuStage{}, PackOut{});

        cb_push_back(cb_out, num_tiles_per_cycle);

        // Pop non-broadcast CBs inside loop
#if !BCAST_A
        cb_pop_front(predicate_cb, num_tiles_per_cycle);
#endif
#if !BCAST_B
        cb_pop_front(true_cb, num_tiles_per_cycle);
#endif
#if !BCAST_C
        cb_pop_front(false_cb, num_tiles_per_cycle);
#endif
    }

    // Pop broadcast CBs outside loop
#if BCAST_A
    cb_pop_front(predicate_cb, num_tiles_per_cycle);
#endif
#if BCAST_B
    cb_pop_front(true_cb, num_tiles_per_cycle);
#endif
#if BCAST_C
    cb_pop_front(false_cb, num_tiles_per_cycle);
#endif
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto predicate_cb = tt::CBIndex::c_0;
    constexpr auto true_cb = tt::CBIndex::c_1;
    constexpr auto false_cb = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(predicate_cb, cb_out);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile<predicate_cb, true_cb, false_cb, cb_out>(tile_freq, tile_start, num_tiles_per_cycle);
    }

    if (remaining_iterations > 0) {
        process_tile<predicate_cb, true_cb, false_cb, cb_out>(remaining_iterations, tile_start, num_tiles_per_cycle);
    }
}
