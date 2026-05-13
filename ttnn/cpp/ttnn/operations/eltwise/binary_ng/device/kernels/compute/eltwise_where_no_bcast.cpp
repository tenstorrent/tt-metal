// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/fill.h"
#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {
// =============================================================================
// Custom block elements for Where(cond, true, false) with stride-3 DEST scratch:
//   D[3j]   = cond (cb_in0)
//   D[3j+1] = true value  (tensor for TTS, scalar for TST)
//   D[3j+2] = false value (scalar for TTS, tensor for TST)
// Final BINARY_SFPU_OP stores result to D[3j].
//
// The middle stage (host-injected FILL_LLK + copy + BINARY_SFPU_OP) is encapsulated
// as a single DEST-only chain element ("LocalWhereStage").
// =============================================================================

template <uint32_t Cb, uint32_t BlockSize>
struct BlockCopyTileStride3Cond : compute_kernel_lib::CopyTileTag {
    static constexpr uint32_t cb = Cb;
    static constexpr uint32_t cb_a_id() { return Cb; }
    static constexpr uint32_t cb_b_id() { return 0; }
    static constexpr compute_kernel_lib::Dst dst_slot = compute_kernel_lib::Dst::D0;
    static constexpr compute_kernel_lib::CopyTilePolicy a_policy() {
        return compute_kernel_lib::CopyTilePolicy::WaitAndPop;
    }
    static constexpr compute_kernel_lib::CopyTilePolicy b_policy() {
        return compute_kernel_lib::CopyTilePolicy::NoWaitNoPop;
    }
    static constexpr bool is_upfront = false;
    static constexpr bool clashes_with_fpu = true;
    static constexpr uint32_t block_size = BlockSize;

    static ALWI void init() { copy_tile_to_dst_init_short(Cb); }
    ALWI void wait_per_tile(uint32_t /*i*/) const { cb_wait_front(Cb, BlockSize); }
    ALWI void wait_upfront(uint32_t /*n*/) const {}
    ALWI void exec(uint32_t /*i*/, uint32_t /*slot_offset*/) const {
        for (uint32_t j = 0; j < BlockSize; ++j) {
            copy_tile(Cb, j, j * 3);
        }
    }
    ALWI void pop_per_tile(uint32_t /*i*/) const { cb_pop_front(Cb, BlockSize); }
    ALWI void pop_upfront_end(uint32_t /*n*/) const {}
};

// Per-iter:
//   copy tensor from CbTensor to D[3j+1] (TTS) or D[3j+2] (TST)
//   fill scalar at D[3j+2] (TTS) or D[3j+1] (TST)
//   BINARY_SFPU_OP(3j, 3j+1, 3j+2, 3j)
template <uint32_t CbTensor, uint32_t BlockSize>
struct LocalWhereStage : compute_kernel_lib::CopyTileTag {
    // CB-reader trait surface so chain dispatcher invokes member exec(i).
    static constexpr uint32_t cb = CbTensor;
    static constexpr uint32_t cb_a_id() { return CbTensor; }
    static constexpr uint32_t cb_b_id() { return 0; }
    static constexpr compute_kernel_lib::Dst dst_slot = compute_kernel_lib::Dst::D1;
    static constexpr compute_kernel_lib::CopyTilePolicy a_policy() {
        return compute_kernel_lib::CopyTilePolicy::WaitAndPop;
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
    ALWI void wait_per_tile(uint32_t /*i*/) const { cb_wait_front(CbTensor, BlockSize); }
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
    ALWI void pop_per_tile(uint32_t /*i*/) const { cb_pop_front(CbTensor, BlockSize); }
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
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);
    const auto scalar_val = reinterpret_cast<const float*>(&scalar_value);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    unary_op_init_common(cb_in0, cb_out);
    BINARY_SFPU_INIT

    using CondLoad = BlockCopyTileStride3Cond<cb_in0, num_tiles_per_cycle>;
    using WhereMid = LocalWhereStage<cb_in1, num_tiles_per_cycle>;
    using PackStage = BlockPackTileStride3<cb_out, num_tiles_per_cycle>;

    WhereMid mid{};
    mid.scalar_value = scalar_value;
    mid.scalar_val = scalar_val;

    eltwise_chain(num_tiles, CondLoad{}, mid, PackStage{});
}
