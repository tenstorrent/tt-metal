// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/div_int32_floor.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/binary_remainder.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/quantization.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/atan2.h"
#include "api/compute/binary_comp.h"

#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_block.hpp"

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
namespace {
// =============================================================================
// Custom block elements — stride-2 DEST scratch (lhs in 2j, rhs in 2j+1, out in 2j).
//
// These match compute_kernel_lib's chain element trait surface (CopyTileTag /
// DestOnlyTag / PackTileTag) and can be composed via eltwise_chain().
// =============================================================================

template <uint32_t Cb, uint32_t OldCb, uint32_t BlockSize>
struct BlockCopyTileStride2Lhs : compute_kernel_lib::CopyTileTag {
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

    static ALWI void init() { copy_tile_to_dst_init_short_with_dt(OldCb, Cb); }

    ALWI void wait_per_tile(uint32_t /*i*/) const { cb_wait_front(Cb, BlockSize); }
    ALWI void wait_upfront(uint32_t /*n*/) const {}
    ALWI void exec(uint32_t /*i*/) const {
        for (uint32_t j = 0; j < BlockSize; ++j) {
            copy_tile(Cb, j, j * 2);
        }
    }
    ALWI void pop_per_tile(uint32_t /*i*/) const { cb_pop_front(Cb, BlockSize); }
    ALWI void pop_upfront_end(uint32_t /*n*/) const {}
};

template <uint32_t Cb, uint32_t OldCb, uint32_t BlockSize>
struct BlockCopyTileStride2Rhs : compute_kernel_lib::CopyTileTag {
    static constexpr uint32_t cb = Cb;
    static constexpr uint32_t cb_a_id() { return Cb; }
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

    static ALWI void init() { copy_tile_to_dst_init_short_with_dt(OldCb, Cb); }

    ALWI void wait_per_tile(uint32_t /*i*/) const { cb_wait_front(Cb, BlockSize); }
    ALWI void wait_upfront(uint32_t /*n*/) const {}
    ALWI void exec(uint32_t /*i*/) const {
        for (uint32_t j = 0; j < BlockSize; ++j) {
            copy_tile(Cb, j, j * 2 + 1);
        }
    }
    ALWI void pop_per_tile(uint32_t /*i*/) const { cb_pop_front(Cb, BlockSize); }
    ALWI void pop_upfront_end(uint32_t /*n*/) const {}
};

// Local SFPU stage — DEST-only chain element wrapping host-injected BINARY_SFPU_OP macro.
// Per chain dispatcher: DestOnly (non-Fill/Rand) elements expose static exec().
template <uint32_t BlockSize>
struct LocalBlockSfpuBinary : compute_kernel_lib::DestOnlyTag {
    static constexpr bool is_upfront = false;
    static constexpr bool clashes_with_fpu = false;
    static constexpr uint32_t block_size = BlockSize;

    static ALWI void init() { BINARY_SFPU_INIT; }
    static ALWI void exec() {
        for (uint32_t j = 0; j < BlockSize; ++j) {
            BINARY_SFPU_OP(j * 2, j * 2 + 1, j * 2);
        }
    }
};

// Pack from stride-2 DEST: dst index 2j, output cb index j.
template <uint32_t Cb, uint32_t BlockSize>
struct BlockPackTileStride2 : compute_kernel_lib::PackTileTag {
    static constexpr uint32_t cb = Cb;
    static constexpr uint32_t pack_cb_id() { return Cb; }
    static constexpr compute_kernel_lib::Dst pack_dst_slot = compute_kernel_lib::Dst::D0;
    static constexpr bool is_upfront = false;
    static constexpr uint32_t block_size = BlockSize;

    static ALWI void init() {}
    ALWI void reserve_per_tile(uint32_t /*i*/) const { cb_reserve_back(Cb, BlockSize); }
    ALWI void reserve_upfront(uint32_t /*n*/) const {}
    ALWI void exec(uint32_t /*i*/) const {
        for (uint32_t j = 0; j < BlockSize; ++j) {
            pack_tile(j * 2, Cb, j);
        }
    }
    ALWI void push_per_tile(uint32_t /*i*/) const { cb_push_back(Cb, BlockSize); }
    ALWI void push_at_end(uint32_t /*n*/) const {}
};
}  // namespace
#endif  // no-activations

FORCE_INLINE void process_sfpu_tiles(
    uint32_t n, uint32_t cb_pre_lhs, uint32_t cb_post_lhs, uint32_t cb_pre_rhs, uint32_t cb_post_rhs, uint32_t cb_out) {
    PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, n);
    cb_wait_front(cb_post_lhs, n);

    PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, n);
    cb_wait_front(cb_post_rhs, n);

    cb_reserve_back(cb_out, n);

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT;
#endif

    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
    for (uint32_t i = 0; i < n; ++i) {
        copy_tile(cb_post_lhs, i, i * 2);
    }
    copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
    for (uint32_t i = 0; i < n; ++i) {
        copy_tile(cb_post_rhs, i, i * 2 + 1);
#if HAS_ACTIVATIONS(POST)
        BINARY_SFPU_INIT;
#endif
        BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);
        PROCESS_POST_ACTIVATIONS(i * 2);
    }
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile(i * 2, cb_out);
    }
    tile_regs_release();

    cb_push_back(cb_out, n);
    cb_pop_front(cb_post_lhs, n);
    cb_pop_front(cb_post_rhs, n);
}

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    unary_op_init_common(cb_post_lhs, cb_out);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
    // No-activations fast path — block-mode chain with stride-2 DEST scratch.
    using LhsLoad = BlockCopyTileStride2Lhs<(uint32_t)cb_post_lhs, (uint32_t)cb_post_rhs, num_tiles_per_cycle>;
    using RhsLoad = BlockCopyTileStride2Rhs<(uint32_t)cb_post_rhs, (uint32_t)cb_post_lhs, num_tiles_per_cycle>;
    using SfpuStage = LocalBlockSfpuBinary<num_tiles_per_cycle>;
    using PackStage = BlockPackTileStride2<(uint32_t)cb_out, num_tiles_per_cycle>;

    const uint32_t num_blocks = num_tiles / num_tiles_per_cycle;
    eltwise_chain(num_blocks, LhsLoad{}, RhsLoad{}, SfpuStage{}, PackStage{});

    // Remainder — keep raw inline (matches reference partials).
    uint32_t remainder = num_tiles % num_tiles_per_cycle;
    if (remainder > 0) {
        process_sfpu_tiles(remainder, cb_pre_lhs, cb_post_lhs, cb_pre_rhs, cb_post_rhs, cb_out);
    }
#else
    // Activations path — keep raw (chain doesn't yet wrap PREPROCESS / PROCESS_POST_ACTIVATIONS).
    // Process full chunks
    uint32_t num_full_chunks = num_tiles / num_tiles_per_cycle;
    for (uint32_t chunk = 0; chunk < num_full_chunks; ++chunk) {
        process_sfpu_tiles(num_tiles_per_cycle, cb_pre_lhs, cb_post_lhs, cb_pre_rhs, cb_post_rhs, cb_out);
    }

    // Process remainder
    uint32_t remainder = num_tiles % num_tiles_per_cycle;
    if (remainder > 0) {
        process_sfpu_tiles(remainder, cb_pre_lhs, cb_post_lhs, cb_pre_rhs, cb_post_rhs, cb_out);
    }
#endif
}
