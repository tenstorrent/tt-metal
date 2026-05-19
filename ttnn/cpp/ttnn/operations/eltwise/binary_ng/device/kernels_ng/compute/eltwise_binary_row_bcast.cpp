// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils.hpp"
#include "experimental/circular_buffer.h"

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

    binary_op_init_common(cb_post_lhs, cb_post_rhs, cb_out);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // ----- LLK row-broadcast preamble (kept raw) -----
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
        pack_reconfig_data_format(cb_llk_post, cb_out);
#ifdef ARCH_BLACKHOLE
        PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(cb_out)));
#endif

        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        exp_cb_post_lhs.wait_front(num_tiles_per_cycle);

        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        exp_cb_post_rhs.wait_front(num_tiles_per_cycle);

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
        // ---- Migrated stage: chunked-streaming BinaryFpu + PackTile.
        //      CB lifecycle for inputs is owned by the outer scope (PREPROCESS / cb_post_*).
        //      PerBlockReserveAndPush coalesces num_tiles_per_cycle reserve/push into a
        //      single pair (vs. per-tile reserve/push at BlockSize=1).
        using BinElt = BinaryFpu<
            cb_post_lhs,
            cb_post_rhs,
            FPU_OP,
            BroadcastDim::None,
            BinaryDataFormatReconfig::None,
            CopyTilePolicy::NoWaitNoPop,
            CopyTilePolicy::NoWaitNoPop,
            CbIndexMode::BlockIter,
            Dst::D0,
            CbIndexMode::BlockIter>;
        using PackElt = PackTile<
            cb_out,
            Dst::D0,
            PackTilePolicy::PerBlockReserveAndPush,
            PackTileIndexMode::BlockIter,
            PackTileReconfig::None>;
        eltwise_chain<num_tiles_per_cycle>(num_tiles_per_cycle, BinElt{}, PackElt{});
#else
        // Activations path — keep raw (PROCESS_POST_ACTIVATIONS macro injection).
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
        exp_cb_out.reserve_back(num_tiles_per_cycle);

        tile_regs_acquire();
        BINARY_OP(cb_post_lhs, cb_post_rhs, 0, 0, 0);
        PROCESS_POST_ACTIVATIONS(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        exp_cb_out.push_back(num_tiles_per_cycle);
#endif
        exp_cb_post_lhs.pop_front(num_tiles_per_cycle);
        exp_cb_post_rhs.pop_front(num_tiles_per_cycle);
    }
}
