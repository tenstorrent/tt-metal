// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {
// =============================================================================
// Where(cond, true_val, false_val) with stride-3 DEST scratch.
// One operand is the broadcast input (waited/popped at outer level), the other
// is per-iter (waited/popped per chain iteration).
// =============================================================================

template <uint32_t Cb, uint32_t BlockSize, bool DoWaitPop>
struct BlockCopyTileStride3In0 : compute_kernel_lib::CopyTileTag {
    static constexpr uint32_t cb = Cb;
    static constexpr uint32_t cb_a_id() { return Cb; }
    static constexpr uint32_t cb_b_id() { return 0; }
    static constexpr compute_kernel_lib::Dst dst_slot = compute_kernel_lib::Dst::D0;
    static constexpr compute_kernel_lib::CopyTilePolicy a_policy() {
        return DoWaitPop ? compute_kernel_lib::CopyTilePolicy::WaitAndPop
                         : compute_kernel_lib::CopyTilePolicy::NoWaitNoPop;
    }
    static constexpr compute_kernel_lib::CopyTilePolicy b_policy() {
        return compute_kernel_lib::CopyTilePolicy::NoWaitNoPop;
    }
    static constexpr bool is_upfront = false;
    static constexpr bool clashes_with_fpu = true;
    static constexpr uint32_t block_size = BlockSize;

    static ALWI void init() { copy_tile_to_dst_init_short(Cb); }
    ALWI void wait_per_tile(uint32_t /*i*/) const {
        if constexpr (DoWaitPop) {
            cb_wait_front(Cb, BlockSize);
        }
    }
    ALWI void wait_upfront(uint32_t /*n*/) const {}
    ALWI void exec(uint32_t /*i*/, uint32_t /*slot_offset*/) const {
        for (uint32_t j = 0; j < BlockSize; ++j) {
            copy_tile(Cb, j, j * 3);
        }
    }
    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (DoWaitPop) {
            cb_pop_front(Cb, BlockSize);
        }
    }
    ALWI void pop_upfront_end(uint32_t /*n*/) const {}
};

template <uint32_t CbTensor, uint32_t BlockSize, bool DoWaitPop>
struct LocalWhereStage : compute_kernel_lib::CopyTileTag {
    static constexpr uint32_t cb = CbTensor;
    static constexpr uint32_t cb_a_id() { return CbTensor; }
    static constexpr uint32_t cb_b_id() { return 0; }
    static constexpr compute_kernel_lib::Dst dst_slot = compute_kernel_lib::Dst::D1;
    static constexpr compute_kernel_lib::CopyTilePolicy a_policy() {
        return DoWaitPop ? compute_kernel_lib::CopyTilePolicy::WaitAndPop
                         : compute_kernel_lib::CopyTilePolicy::NoWaitNoPop;
    }
    static constexpr compute_kernel_lib::CopyTilePolicy b_policy() {
        return compute_kernel_lib::CopyTilePolicy::NoWaitNoPop;
    }
    static constexpr bool is_upfront = false;
    static constexpr bool clashes_with_fpu = true;
    static constexpr uint32_t block_size = BlockSize;

    uint32_t scalar_value = 0;
    const float* scalar_val = nullptr;

    static ALWI void init() { copy_tile_to_dst_init_short(CbTensor); }
    ALWI void wait_per_tile(uint32_t /*i*/) const {
        if constexpr (DoWaitPop) {
            cb_wait_front(CbTensor, BlockSize);
        }
    }
    ALWI void wait_upfront(uint32_t /*n*/) const {}
    ALWI void exec(uint32_t /*i*/, uint32_t /*slot_offset*/) const {
        for (uint32_t j = 0; j < BlockSize; ++j) {
#if WHERE_TTS
            copy_tile(CbTensor, j, j * 3 + 1);
            fill_tile_init();
#ifdef FILL_WITH_VALUE_FLOAT
            FILL_LLK(j * 3 + 2, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
            FILL_LLK(j * 3 + 2, scalar_value);
#endif
#endif
#if WHERE_TST
            copy_tile(CbTensor, j, j * 3 + 2);
            fill_tile_init();
#ifdef FILL_WITH_VALUE_FLOAT
            FILL_LLK(j * 3 + 1, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
            FILL_LLK(j * 3 + 1, scalar_value);
#endif
#endif
            BINARY_SFPU_OP(j * 3, j * 3 + 1, j * 3 + 2, j * 3);
        }
    }
    ALWI void pop_per_tile(uint32_t /*i*/) const {
        if constexpr (DoWaitPop) {
            cb_pop_front(CbTensor, BlockSize);
        }
    }
    ALWI void pop_upfront_end(uint32_t /*n*/) const {}
};

template <uint32_t Cb, uint32_t BlockSize>
struct BlockPackTileStride3 : compute_kernel_lib::PackTileTag {
    static constexpr uint32_t cb = Cb;
    static constexpr uint32_t pack_cb_id() { return Cb; }
    static constexpr compute_kernel_lib::Dst pack_dst_slot = compute_kernel_lib::Dst::D0;
    static constexpr bool is_upfront = false;
    static constexpr uint32_t block_size = BlockSize;

    static ALWI void init() {}
    ALWI void reserve_per_tile(uint32_t /*i*/) const { cb_reserve_back(Cb, BlockSize); }
    ALWI void reserve_upfront(uint32_t /*n*/) const {}
    ALWI void exec(uint32_t /*i*/, uint32_t /*slot_offset*/) const {
        for (uint32_t j = 0; j < BlockSize; ++j) {
            pack_tile(j * 3, Cb, j);
        }
    }
    ALWI void push_per_tile(uint32_t /*i*/) const { cb_push_back(Cb, BlockSize); }
    ALWI void push_at_end(uint32_t /*n*/) const {}
};
}  // namespace

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);
    const float* scalar_value_ptr = reinterpret_cast<const float*>(&scalar_value);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    unary_op_init_common(cb_in0, cb_out);
    BINARY_SFPU_INIT

#if BCAST_INPUT
    // BCAST_INPUT=1 : cb_in1 is broadcast; cb_in0 (condition) waited/popped per-iter.
    constexpr uint32_t cb_bcast = (uint32_t)cb_in1;
    constexpr bool wait_pop_in0 = true;
    constexpr bool wait_pop_in1 = false;
#else
    // BCAST_INPUT=0 : cb_in0 (condition) is broadcast; cb_in1 (true/false tensor) waited/popped per-iter.
    constexpr uint32_t cb_bcast = (uint32_t)cb_in0;
    constexpr bool wait_pop_in0 = false;
    constexpr bool wait_pop_in1 = true;
#endif

    using CondLoad = BlockCopyTileStride3In0<cb_in0, num_tiles_per_cycle, wait_pop_in0>;
    using WhereMid = LocalWhereStage<cb_in1, num_tiles_per_cycle, wait_pop_in1>;
    using PackStage = BlockPackTileStride3<cb_out, num_tiles_per_cycle>;

    WhereMid mid{};
    mid.scalar_value = scalar_value;
    mid.scalar_val = scalar_value_ptr;

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    auto run_iter = [&](uint32_t freq, uint32_t start) {
        cb_wait_front(cb_bcast, num_tiles_per_cycle);
        eltwise_chain(freq - start, CondLoad{}, mid, PackStage{});
        cb_pop_front(cb_bcast, num_tiles_per_cycle);
    };

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        run_iter(tile_freq, tile_start);
    }
    if (remaining_iterations > 0) {
        run_iter(remaining_iterations, tile_start);
    }
}
