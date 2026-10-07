// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#ifndef BINARY_NG_BLOCK
#define BINARY_NG_BLOCK 0
#endif
#ifndef BINARY_NG_BLOCK_PACK
#define BINARY_NG_BLOCK_PACK 0
#endif
// Blackhole: ELWMUL, which binary_ng runs at HiFi4, takes the per-tile hand-off; add and sub keep the per-face one, except
// in the block section (BINARY_NG_BLOCK), whose block unpack takes it for every op.
#ifndef EB_R3_PER_FACE
#define EB_R3_PER_FACE 0
#endif
#ifndef EB_R3_MULTI_PASS
#define EB_R3_MULTI_PASS 0
#endif
#ifndef EB_R3_PROBE_CHUNK_INIT
#define EB_R3_PROBE_CHUNK_INIT 0
#endif
#define ELTWISE_BINARY_PER_TILE_HANDOFF ((BINARY_OP_TYPE == EltwiseBinaryType::ELWMUL || BINARY_NG_BLOCK) && !EB_R3_PER_FACE)
#define ELTWISE_BINARY_BLOCK_UNPACK BINARY_NG_BLOCK
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "eltwise_utils_common.hpp"
#include "eltwise_utils.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    constexpr auto cb_pre_lhs_id = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs_id = tt::CBIndex::c_1;

    constexpr auto cb_post_lhs_id = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs_id;
    static_assert(
        cb_post_lhs_id == BINARY_FPU_SRCA_FORMAT_CB,
        "binary_ng: FPU SrcA startup operand disagrees with the preprocessing restore reference");
    CircularBuffer cb_post_lhs(cb_post_lhs_id);
    CircularBuffer cb_post_rhs(HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs_id);
    CircularBuffer cb_out(tt::CBIndex::c_2);

    compute_kernel_hw_startup(cb_post_lhs.get_cb_id(), cb_post_rhs.get_cb_id(), cb_out.get_cb_id());
#ifdef PACK_RELU
    pack_relu_config(ReluConfig::zero());
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or BINARY_POST_REINIT)
    binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs.get_cb_id(), cb_post_rhs.get_cb_id());
#endif

    // Inline helper to process n tiles
#if EB_R3_MULTI_PASS
    auto pre_tiles = [&](uint32_t n) {
        PREPROCESS(LHS, CircularBuffer(cb_pre_lhs_id), cb_post_lhs, cb_out, n);
        PREPROCESS(RHS, CircularBuffer(cb_pre_rhs_id), cb_post_rhs, cb_out, n);
    };
#endif
    auto process_tiles = [&](uint32_t n) {
#if !EB_R3_MULTI_PASS
        PREPROCESS(LHS, CircularBuffer(cb_pre_lhs_id), cb_post_lhs, cb_out, n);
        cb_post_lhs.wait_front(n);

        PREPROCESS(RHS, CircularBuffer(cb_pre_rhs_id), cb_post_rhs, cb_out, n);
        cb_post_rhs.wait_front(n);
#else
        cb_post_lhs.wait_front(n);
        cb_post_rhs.wait_front(n);
#endif

        cb_out.reserve_back(n);

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or BINARY_POST_REINIT) && !EB_R3_MULTI_PASS
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs.get_cb_id(), cb_post_rhs.get_cb_id());
#if EB_R3_PROBE_CHUNK_INIT
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs.get_cb_id(), cb_post_rhs.get_cb_id());
#endif
#endif
        tile_regs_acquire();
#if BINARY_NG_BLOCK
        if constexpr (BINARY_OP_TYPE == EltwiseBinaryType::ELWMUL) {
            mul_block(cb_post_lhs.get_cb_id(), cb_post_rhs.get_cb_id(), 0, 0, 0, n);
        } else if constexpr (BINARY_OP_TYPE == EltwiseBinaryType::ELWADD) {
            add_block(cb_post_lhs.get_cb_id(), cb_post_rhs.get_cb_id(), 0, 0, 0, n);
        } else {
            sub_block(cb_post_lhs.get_cb_id(), cb_post_rhs.get_cb_id(), 0, 0, 0, n);
        }
        for (uint32_t i = 0; i < n; ++i) {
            PROCESS_POST_ACTIVATIONS(i);
        }
#else
        for (uint32_t i = 0; i < n; ++i) {
            BINARY_OP(cb_post_lhs.get_cb_id(), cb_post_rhs.get_cb_id(), i, i, i);
            PROCESS_POST_ACTIVATIONS(i);
        }
#endif
        tile_regs_commit();

        tile_regs_wait();
#if BINARY_NG_BLOCK_PACK
        pack_block_mop(0, cb_out.get_cb_id(), n);
#else
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_out.get_cb_id());
        }
#endif
        tile_regs_release();

        cb_out.push_back(n);
        cb_post_lhs.pop_front(n);
        cb_post_rhs.pop_front(n);
    };

    uint32_t num_full_chunks = num_tiles / num_tiles_per_cycle;
    uint32_t remainder = num_tiles % num_tiles_per_cycle;
#if EB_R3_MULTI_PASS
    // CI only (#58725): the operand pass over up to EB_R3_MULTI_PASS sections, then one binary init for all of them
    const uint32_t num_chunks = num_full_chunks + (remainder > 0);
    for (uint32_t g = 0; g < num_chunks; g += EB_R3_MULTI_PASS) {
        const uint32_t m = num_chunks - g < EB_R3_MULTI_PASS ? num_chunks - g : EB_R3_MULTI_PASS;
        for (uint32_t j = 0; j < m; ++j) {
            pre_tiles(g + j < num_full_chunks ? num_tiles_per_cycle : remainder);
        }
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs.get_cb_id(), cb_post_rhs.get_cb_id());
        for (uint32_t j = 0; j < m; ++j) {
            process_tiles(g + j < num_full_chunks ? num_tiles_per_cycle : remainder);
        }
    }
#else
    // Process full chunks
    for (uint32_t chunk = 0; chunk < num_full_chunks; ++chunk) {
        process_tiles(num_tiles_per_cycle);
    }

    // Process remainder
    if (remainder > 0) {
        process_tiles(remainder);
    }
#endif
}
