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
#include "api/compute/binary_fmod.h"
#include "api/compute/binary_remainder.h"
#include "api/compute/quantization.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/binary_comp.h"
#include "api/compute/atan2.h"
#include "api/compute/bcast.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils.hpp"
#include "experimental/circular_buffer.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_block.hpp"

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
namespace {
// Stride-2 DEST scratch chain elements (lhs in 2j, rhs in 2j+1, out from 2j).
template <uint32_t Cb, uint32_t OldCb, uint32_t BlockSize>
struct BlockCopyTileStride2Lhs : compute_kernel_lib::CopyTileTag {
    static constexpr uint32_t cb = Cb;
    static constexpr uint32_t cb_a_id() { return Cb; }
    static constexpr uint32_t cb_b_id() { return 0; }
    static constexpr compute_kernel_lib::Dst dst_slot = compute_kernel_lib::Dst::D0;
    static constexpr compute_kernel_lib::CopyTilePolicy a_policy() {
        return compute_kernel_lib::CopyTilePolicy::NoWaitNoPop;  // outer scope handles wait/pop
    }
    static constexpr compute_kernel_lib::CopyTilePolicy b_policy() {
        return compute_kernel_lib::CopyTilePolicy::NoWaitNoPop;
    }
    static constexpr bool is_upfront = false;
    static constexpr bool clashes_with_fpu = true;
    static constexpr uint32_t block_size = BlockSize;

    static ALWI void init() { copy_tile_to_dst_init_short_with_dt(OldCb, Cb); }
    ALWI void wait_per_tile(uint32_t /*i*/) const {}
    ALWI void wait_upfront(uint32_t /*n*/) const {}
    ALWI void exec(uint32_t /*i*/) const {
        for (uint32_t j = 0; j < BlockSize; ++j) copy_tile(Cb, j, j * 2);
    }
    ALWI void pop_per_tile(uint32_t /*i*/) const {}
    ALWI void pop_upfront_end(uint32_t /*n*/) const {}
};

template <uint32_t Cb, uint32_t OldCb, uint32_t BlockSize>
struct BlockCopyTileStride2Rhs : compute_kernel_lib::CopyTileTag {
    static constexpr uint32_t cb = Cb;
    static constexpr uint32_t cb_a_id() { return Cb; }
    static constexpr uint32_t cb_b_id() { return 0; }
    static constexpr compute_kernel_lib::Dst dst_slot = compute_kernel_lib::Dst::D1;
    static constexpr compute_kernel_lib::CopyTilePolicy a_policy() {
        return compute_kernel_lib::CopyTilePolicy::NoWaitNoPop;
    }
    static constexpr compute_kernel_lib::CopyTilePolicy b_policy() {
        return compute_kernel_lib::CopyTilePolicy::NoWaitNoPop;
    }
    static constexpr bool is_upfront = false;
    static constexpr bool clashes_with_fpu = true;
    static constexpr uint32_t block_size = BlockSize;

    static ALWI void init() { copy_tile_to_dst_init_short_with_dt(OldCb, Cb); }
    ALWI void wait_per_tile(uint32_t /*i*/) const {}
    ALWI void wait_upfront(uint32_t /*n*/) const {}
    ALWI void exec(uint32_t /*i*/) const {
        for (uint32_t j = 0; j < BlockSize; ++j) copy_tile(Cb, j, j * 2 + 1);
    }
    ALWI void pop_per_tile(uint32_t /*i*/) const {}
    ALWI void pop_upfront_end(uint32_t /*n*/) const {}
};

template <uint32_t BlockSize>
struct LocalBlockSfpuBinary : compute_kernel_lib::DestOnlyTag {
    static constexpr bool is_upfront = false;
    static constexpr bool clashes_with_fpu = false;
    static constexpr uint32_t block_size = BlockSize;
    static ALWI void init() {}  // BINARY_SFPU_INIT done at kernel top
    static ALWI void exec() {
        for (uint32_t j = 0; j < BlockSize; ++j) {
            BINARY_SFPU_OP(j * 2, j * 2 + 1, j * 2);
        }
    }
};

template <uint32_t Cb, uint32_t BlockSize>
struct BlockPackTileStride2 : compute_kernel_lib::PackTileTag {
    static constexpr uint32_t cb = Cb;
    static constexpr uint32_t pack_cb_id() { return Cb; }
    static constexpr compute_kernel_lib::Dst pack_dst_slot = compute_kernel_lib::Dst::D0;
    static constexpr bool is_upfront = false;
    static constexpr uint32_t block_size = BlockSize;
    static ALWI void init() {}
    ALWI void reserve_per_tile(uint32_t /*i*/) const {}  // outer scope reserves
    ALWI void reserve_upfront(uint32_t /*n*/) const {}
    ALWI void exec(uint32_t /*i*/) const {
        for (uint32_t j = 0; j < BlockSize; ++j) pack_tile(j * 2, Cb, j);
    }
    ALWI void push_per_tile(uint32_t /*i*/) const {}  // outer scope pushes
    ALWI void push_at_end(uint32_t /*n*/) const {}
};
}  // namespace
#endif

void kernel_main() {
    using namespace compute_kernel_lib;
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    constexpr auto cb_out = tt::CBIndex::c_2;
    experimental::CircularBuffer exp_cb_out(cb_out);

#if SRC_BCAST
    constexpr auto cb_bcast = tt::CBIndex::c_0;
    constexpr auto cb_llk_post = tt::CBIndex::c_5;
    constexpr auto cb_pre_lhs = cb_llk_post;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_llk_post;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : tt::CBIndex::c_1;
#endif
#if SRC_BCAST_B
    constexpr auto cb_bcast = tt::CBIndex::c_1;
    constexpr auto cb_llk_post = tt::CBIndex::c_6;
    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = cb_llk_post;
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : tt::CBIndex::c_0;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_llk_post;
#endif

    experimental::CircularBuffer exp_cb_bcast(cb_bcast);
    experimental::CircularBuffer exp_cb_llk_post(cb_llk_post);
    experimental::CircularBuffer exp_cb_post_lhs(cb_post_lhs);
    experimental::CircularBuffer exp_cb_post_rhs(cb_post_rhs);

    unary_op_init_common(cb_post_lhs, cb_out);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
#endif

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        exp_cb_bcast.wait_front(num_tiles_per_cycle);
        exp_cb_llk_post.reserve_back(num_tiles_per_cycle);
        unary_bcast_init<BroadcastType::ROW>(cb_bcast, cb_llk_post);

        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_bcast, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_llk_post);
        exp_cb_llk_post.push_back(num_tiles_per_cycle);
        tile_regs_release();
        exp_cb_bcast.pop_front(num_tiles_per_cycle);
        // unary_bcast_uninit<BroadcastType::ROW>(cb_bcast);
        pack_reconfig_data_format(cb_llk_post, cb_out);
#ifdef ARCH_BLACKHOLE
        PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(cb_out)));
#endif

        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        exp_cb_post_lhs.wait_front(num_tiles_per_cycle);

        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        exp_cb_post_rhs.wait_front(num_tiles_per_cycle);

        exp_cb_out.reserve_back(num_tiles_per_cycle);
#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        // Migrated stage: stride-2 DEST scratch chain.
        using LhsLoad = BlockCopyTileStride2Lhs<cb_post_lhs, cb_post_rhs, num_tiles_per_cycle>;
        using RhsLoad = BlockCopyTileStride2Rhs<cb_post_rhs, cb_post_lhs, num_tiles_per_cycle>;
        using SfpuStage = LocalBlockSfpuBinary<num_tiles_per_cycle>;
        using PackStage = BlockPackTileStride2<cb_out, num_tiles_per_cycle>;
        eltwise_chain(1u, LhsLoad{}, RhsLoad{}, SfpuStage{}, PackStage{});
#else
#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT
#endif
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_lhs, i, i * 2);
        }
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_rhs, i, i * 2 + 1);

#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT
#endif
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);
            PROCESS_POST_ACTIVATIONS(i * 2);
        }
        tile_regs_commit();

        tile_regs_wait();

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, cb_out);
        }
        tile_regs_release();
#endif

        exp_cb_out.push_back(num_tiles_per_cycle);
        exp_cb_post_lhs.pop_front(num_tiles_per_cycle);
        exp_cb_post_rhs.pop_front(num_tiles_per_cycle);
    }
}
