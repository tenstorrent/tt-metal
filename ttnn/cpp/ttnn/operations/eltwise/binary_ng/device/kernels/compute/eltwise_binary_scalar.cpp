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
#ifndef BINARY_NG_PRE_SECTIONS
#define BINARY_NG_PRE_SECTIONS 0
#endif
// Blackhole: ELWMUL, which binary_ng runs at HiFi4, takes the per-tile hand-off; add and sub keep the per-face one, except
// in the block section (BINARY_NG_BLOCK), whose block unpack takes it for every op.
#ifndef EB_R3_PER_FACE
#define EB_R3_PER_FACE 0
#endif
#define ELTWISE_BINARY_PER_TILE_HANDOFF ((BINARY_OP_TYPE == EltwiseBinaryType::ELWMUL || BINARY_NG_BLOCK) && !EB_R3_PER_FACE)
#include "api/compute/eltwise_binary.h"
#if BINARY_NG_BLOCK_PACK
#include "api/compute/experimental/pack_block.h"
#endif

#include "eltwise_utils_common.hpp"
#include "eltwise_utils.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
    // DPRINT("num_tiles_per_cycle: {}\n", num_tiles_per_cycle);
    constexpr auto cb_pre_lhs_id = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs_id = tt::CBIndex::c_1;

    constexpr auto cb_post_lhs_id = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs_id;
    constexpr auto cb_post_rhs_id = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs_id;
    CircularBuffer cb_post_lhs(cb_post_lhs_id);
    CircularBuffer cb_post_rhs(cb_post_rhs_id);
    CircularBuffer cb_out(tt::CBIndex::c_2);

    // FPU operands are unpacked from these CBs straight into srcA/srcB, so the swapped
    // order has to hold for the format setup and the LLK init too, not just the op call.
    // Only the LLK's operand order changes: PREPROCESS and HAS_ACTIVATIONS stay on the physical
    // c_0/c_1 because the host already swapped the activation lists before emitting the defines.
    // Swapping them here as well would apply each activation to the wrong operand.
#if SCALAR_IS_LHS
    static_assert(
        cb_post_rhs_id == BINARY_FPU_SRCA_FORMAT_CB,
        "binary_ng: FPU SrcA startup operand disagrees with the preprocessing restore reference");
    CircularBuffer& cb_op_a = cb_post_rhs;
    CircularBuffer& cb_op_b = cb_post_lhs;
#else
    static_assert(
        cb_post_lhs_id == BINARY_FPU_SRCA_FORMAT_CB,
        "binary_ng: FPU SrcA startup operand disagrees with the preprocessing restore reference");
    CircularBuffer& cb_op_a = cb_post_lhs;
    CircularBuffer& cb_op_b = cb_post_rhs;
#endif

    compute_kernel_hw_startup(cb_op_a.get_cb_id(), cb_op_b.get_cb_id(), cb_out.get_cb_id());
#ifdef PACK_RELU
    pack_relu_config(ReluConfig::zero());
#endif
#if BINARY_NG_BLOCK_PACK
    pack_block_contiguous_init(cb_out.get_cb_id());
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or BINARY_POST_REINIT)
    binary_tiles_init<true, BINARY_OP_TYPE>(cb_op_a.get_cb_id(), cb_op_b.get_cb_id());
#endif

    PREPROCESS(RHS, CircularBuffer(cb_pre_rhs_id), cb_post_rhs, cb_out, 1);
    cb_post_rhs.wait_front(1);

    // Inline lambda to process n tiles with the scalar value
    auto process_tiles = [&](uint32_t n) {
#if !BINARY_NG_PRE_SECTIONS
        PREPROCESS(LHS, CircularBuffer(cb_pre_lhs_id), cb_post_lhs, cb_out, n);
#endif
        cb_post_lhs.wait_front(n);

        cb_out.reserve_back(n);

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or BINARY_POST_REINIT) && !BINARY_NG_PRE_SECTIONS
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_op_a.get_cb_id(), cb_op_b.get_cb_id());
#endif
        tile_regs_acquire();
#if BINARY_NG_BLOCK
        binary_block_strided<BINARY_OP_TYPE>(
            cb_op_a.get_cb_id(), cb_op_b.get_cb_id(), 0, 0, 0, n, SCALAR_IS_LHS ? 0 : 1, SCALAR_IS_LHS ? 1 : 0);
        for (uint32_t i = 0; i < n; ++i) {
            PROCESS_POST_ACTIVATIONS(i);
        }
#else
        for (uint32_t i = 0; i < n; ++i) {
#if SCALAR_IS_LHS
            BINARY_OP(cb_op_a.get_cb_id(), cb_op_b.get_cb_id(), 0, i, i);
#else
            BINARY_OP(cb_op_a.get_cb_id(), cb_op_b.get_cb_id(), i, 0, i);
#endif
            PROCESS_POST_ACTIVATIONS(i);
        }
#endif
        tile_regs_commit();

        tile_regs_wait();
#if BINARY_NG_BLOCK_PACK
        pack_block_contiguous(0, cb_out.get_cb_id(), n);
#else
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_out.get_cb_id());
        }
#endif
        tile_regs_release();

        cb_post_lhs.pop_front(n);
        cb_out.push_back(n);
    };

#if BINARY_NG_PRE_SECTIONS
    // Blackhole: the operand pass runs over BINARY_NG_PRE_SECTIONS sections, then one binary init for all of them
    const uint32_t full_chunks = num_tiles / num_tiles_per_cycle;
    const uint32_t num_chunks = full_chunks + (num_tiles % num_tiles_per_cycle > 0);
    auto chunk_tiles = [&](uint32_t chunk) {
        return chunk < full_chunks ? num_tiles_per_cycle : num_tiles % num_tiles_per_cycle;
    };
    for (uint32_t first = 0; first < num_chunks; first += BINARY_NG_PRE_SECTIONS) {
        const uint32_t last = first + BINARY_NG_PRE_SECTIONS < num_chunks ? first + BINARY_NG_PRE_SECTIONS : num_chunks;
        for (uint32_t chunk = first; chunk < last; ++chunk) {
            PREPROCESS(LHS, CircularBuffer(cb_pre_lhs_id), cb_post_lhs, cb_out, chunk_tiles(chunk));
        }
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_op_a.get_cb_id(), cb_op_b.get_cb_id());
        for (uint32_t chunk = first; chunk < last; ++chunk) {
            process_tiles(chunk_tiles(chunk));
        }
    }
#else
    // Process full chunks
    uint32_t full_chunks = num_tiles / num_tiles_per_cycle;
    for (uint32_t chunk = 0; chunk < full_chunks; ++chunk) {
        process_tiles(num_tiles_per_cycle);
    }

    // Process remainder
    uint32_t remainder = num_tiles % num_tiles_per_cycle;
    if (remainder > 0) {
        process_tiles(remainder);
    }
#endif

    // Pop the scalar tile from RHS CB
    cb_post_rhs.pop_front(1);
}
