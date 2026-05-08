// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary.h"

#include "eltwise_utils_common.hpp"
#include "eltwise_utils.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_block.hpp"

#if BINARY_OP_TYPE == EltwiseBinaryType::ELWADD
constexpr auto FPU_OP = compute_kernel_lib::BinaryFpuOp::Add;
#elif BINARY_OP_TYPE == EltwiseBinaryType::ELWSUB
constexpr auto FPU_OP = compute_kernel_lib::BinaryFpuOp::Sub;
#elif BINARY_OP_TYPE == EltwiseBinaryType::ELWMUL
constexpr auto FPU_OP = compute_kernel_lib::BinaryFpuOp::Mul;
#else
#error "BINARY_OP_TYPE must be ELWADD / ELWSUB / ELWMUL"
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    // D5/D8: caller-side BIG init at the top of MAIN().
    compute_kernel_hw_startup(cb_post_lhs, cb_post_rhs, cb_out);

#if HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST)
    // Activations path — keep raw.
    binary_op_init_common(cb_post_lhs, cb_post_rhs, cb_out);
    PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, 1);
    cb_wait_front(cb_post_rhs, 1);
    auto process_tiles = [&](uint32_t n) {
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, n);
        cb_wait_front(cb_post_lhs, n);
        cb_reserve_back(cb_out, n);
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            BINARY_OP(cb_post_lhs, cb_post_rhs, i, 0, i);
            PROCESS_POST_ACTIVATIONS(i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_out);
        }
        tile_regs_release();
        cb_pop_front(cb_post_lhs, n);
        cb_push_back(cb_out, n);
    };
    uint32_t full_chunks = num_tiles / num_tiles_per_cycle;
    for (uint32_t chunk = 0; chunk < full_chunks; ++chunk) process_tiles(num_tiles_per_cycle);
    uint32_t remainder = num_tiles % num_tiles_per_cycle;
    if (remainder > 0) process_tiles(remainder);
    cb_pop_front(cb_post_rhs, 1);
#else
    // v6 Q4 collapse regression (disposition (c)):
    //
    // The no-activations fast path previously used `BlockBinaryFpu<...,
    // CbIndexMode::BlockIter, CbIndexMode::FirstTile>` (A=block-walk, B=pin scalar
    // tile at idx 0). After the v6 collapse to a single Index template parameter
    // there is no symmetric mode covering both — we revert to the raw LLK shape
    // already used by the activations branch (above). The block-mode optimisation
    // (single acquire/release wrapping the inner block) is preserved.
    binary_op_init_common(cb_post_lhs, cb_post_rhs, cb_out);
    cb_wait_front(cb_post_rhs, 1);
    auto process_tiles_no_act = [&](uint32_t n) {
        cb_wait_front(cb_post_lhs, n);
        cb_reserve_back(cb_out, n);
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            BINARY_OP(cb_post_lhs, cb_post_rhs, i, 0, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_out);
        }
        tile_regs_release();
        cb_pop_front(cb_post_lhs, n);
        cb_push_back(cb_out, n);
    };
    uint32_t full_chunks = num_tiles / num_tiles_per_cycle;
    for (uint32_t chunk = 0; chunk < full_chunks; ++chunk) {
        process_tiles_no_act(num_tiles_per_cycle);
    }
    uint32_t remainder = num_tiles % num_tiles_per_cycle;
    if (remainder > 0) {
        process_tiles_no_act(remainder);
    }
    cb_pop_front(cb_post_rhs, 1);
#endif
}
