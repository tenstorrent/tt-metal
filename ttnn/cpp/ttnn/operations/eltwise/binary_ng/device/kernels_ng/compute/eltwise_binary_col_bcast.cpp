// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace eltwise_binary_kernel_detail {
template <ckernel::EltwiseBinaryType T>
struct FpuOpForBinaryType;
template <>
struct FpuOpForBinaryType<ckernel::EltwiseBinaryType::ELWADD> {
    static constexpr auto value = compute_kernel_lib::BinaryFpuOp::Add;
};
template <>
struct FpuOpForBinaryType<ckernel::EltwiseBinaryType::ELWSUB> {
    static constexpr auto value = compute_kernel_lib::BinaryFpuOp::Sub;
};
template <>
struct FpuOpForBinaryType<ckernel::EltwiseBinaryType::ELWMUL> {
    static constexpr auto value = compute_kernel_lib::BinaryFpuOp::Mul;
};
}  // namespace eltwise_binary_kernel_detail
constexpr auto FPU_OP = eltwise_binary_kernel_detail::FpuOpForBinaryType<ckernel::BINARY_OP_TYPE>::value;

template <
    tt::CBIndex cb_bcast,
    tt::CBIndex cb_llk_post,
    tt::CBIndex cb_pre_lhs,
    tt::CBIndex cb_post_lhs,
    tt::CBIndex cb_pre_rhs,
    tt::CBIndex cb_post_rhs,
    tt::CBIndex cb_out,
    uint32_t num_tiles_per_cycle>
ALWI void process_tile(uint32_t freq, uint32_t tile_start) {
    using namespace ckernel;
    using namespace compute_kernel_lib;

#if BCAST_INPUT
#define CB_PRE_BCAST cb_pre_rhs
#define CB_POST_BCAST cb_post_rhs
#define CB_PRE_OTHER cb_pre_lhs
#define CB_POST_OTHER cb_post_lhs
#else
#define CB_PRE_BCAST cb_pre_lhs
#define CB_POST_BCAST cb_post_lhs
#define CB_PRE_OTHER cb_pre_rhs
#define CB_POST_OTHER cb_post_rhs
#endif
    cb_wait_front(cb_bcast, num_tiles_per_cycle);
    unary_bcast_init<BroadcastType::COL>(cb_bcast, cb_llk_post);
    cb_reserve_back(cb_llk_post, num_tiles_per_cycle);
    tile_regs_acquire();
    unary_bcast<BroadcastType::COL>(cb_bcast, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_llk_post);
    cb_push_back(cb_llk_post, num_tiles_per_cycle);
    tile_regs_release();

    pack_reconfig_data_format(cb_llk_post, cb_out);
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(cb_out)));
#endif

    PREPROCESS(BCAST_OP, CB_PRE_BCAST, CB_POST_BCAST, cb_out, num_tiles_per_cycle);
    cb_wait_front(CB_POST_BCAST, num_tiles_per_cycle);

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
    binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
        PREPROCESS(OTHER_OP, CB_PRE_OTHER, CB_POST_OTHER, cb_out, num_tiles_per_cycle);
        cb_wait_front(CB_POST_OTHER, num_tiles_per_cycle);

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
        // Migrated stage: streaming BinaryFpu + PackTile (auto-block infra).
        cb_reserve_back(cb_out, num_tiles_per_cycle);
        using BinElt = BinaryFpu<
            (uint32_t)cb_post_lhs,
            (uint32_t)cb_post_rhs,
            FPU_OP,
            BroadcastDim::None,
            BinaryDataFormatReconfig::None,
            CopyTilePolicy::NoWaitNoPop,
            CopyTilePolicy::NoWaitNoPop,
            CbIndexMode::BlockIter,
            Dst::D0,
            CbIndexMode::BlockIter>;
        using PackElt =
            PackTile<(uint32_t)cb_out, Dst::D0, PackTilePolicy::NoReserveNoPush, PackTileIndexMode::BlockIter>;
        eltwise_chain<num_tiles_per_cycle>(num_tiles_per_cycle, BinElt{}, PackElt{});
        cb_push_back(cb_out, num_tiles_per_cycle);
#else
        cb_reserve_back(cb_out, num_tiles_per_cycle);
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
        tile_regs_acquire();
        BINARY_OP(cb_post_lhs, cb_post_rhs, 0, 0, 0);
        PROCESS_POST_ACTIVATIONS(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);
#endif
        cb_pop_front(CB_POST_OTHER, num_tiles_per_cycle);
    }
    cb_pop_front(cb_bcast, num_tiles_per_cycle);
    cb_pop_front(CB_POST_BCAST, num_tiles_per_cycle);
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_out = tt::CBIndex::c_2;

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

    binary_op_init_common(cb_post_lhs, cb_post_rhs, cb_out);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
    binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
#endif

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile<
            cb_bcast,
            cb_llk_post,
            cb_pre_lhs,
            cb_post_lhs,
            cb_pre_rhs,
            cb_post_rhs,
            cb_out,
            num_tiles_per_cycle>(tile_freq, tile_start);
    }

    if (remaining_iterations > 0) {
        process_tile<
            cb_bcast,
            cb_llk_post,
            cb_pre_lhs,
            cb_post_lhs,
            cb_pre_rhs,
            cb_post_rhs,
            cb_out,
            num_tiles_per_cycle>(remaining_iterations, tile_start);
    }
}
