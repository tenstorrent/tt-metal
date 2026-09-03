// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary.h"

#include "eltwise_utils_common.hpp"
#include "eltwise_utils.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
    // DPRINT("num_tiles_per_cycle: {}\n", num_tiles_per_cycle);
    constexpr auto cb_pre_lhs_id = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs_id = tt::CBIndex::c_1;

    CircularBuffer cb_post_lhs(HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs_id);
    CircularBuffer cb_post_rhs(HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs_id);
    CircularBuffer cb_out(tt::CBIndex::c_2);

    // FPU operands are unpacked from these CBs straight into srcA/srcB, so the mirrored
    // order has to hold for the format setup and the LLK init too, not just the op call.
    // Only the LLK's view is mirrored: PREPROCESS and HAS_ACTIVATIONS stay on the physical
    // c_0/c_1 because the host already inverted the spans before emitting the defines.
    // Mirroring them here as well would apply each activation to the wrong operand.
#if SCALAR_IS_LHS
    CircularBuffer& cb_op_a = cb_post_rhs;
    CircularBuffer& cb_op_b = cb_post_lhs;
#else
    CircularBuffer& cb_op_a = cb_post_lhs;
    CircularBuffer& cb_op_b = cb_post_rhs;
#endif

    compute_kernel_hw_startup(cb_op_a.get_cb_id(), cb_op_b.get_cb_id(), cb_out.get_cb_id());
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
    binary_tiles_init<true, BINARY_OP_TYPE>(cb_op_a.get_cb_id(), cb_op_b.get_cb_id());
#endif

    PREPROCESS(RHS, CircularBuffer(cb_pre_rhs_id), cb_post_rhs, cb_out, 1);
    cb_post_rhs.wait_front(1);

    // Inline lambda to process n tiles with the scalar value
    auto process_tiles = [&](uint32_t n) {
        PREPROCESS(LHS, CircularBuffer(cb_pre_lhs_id), cb_post_lhs, cb_out, n);
        cb_post_lhs.wait_front(n);

        cb_out.reserve_back(n);

#if HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST)
        binary_tiles_init<true, BINARY_OP_TYPE>(cb_op_a.get_cb_id(), cb_op_b.get_cb_id());
#endif
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
#if SCALAR_IS_LHS
            BINARY_OP(cb_op_a.get_cb_id(), cb_op_b.get_cb_id(), 0, i, i);
#else
            BINARY_OP(cb_op_a.get_cb_id(), cb_op_b.get_cb_id(), i, 0, i);
#endif
            PROCESS_POST_ACTIVATIONS(i);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_out.get_cb_id());
        }
        tile_regs_release();

        cb_post_lhs.pop_front(n);
        cb_out.push_back(n);
    };

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

    // Pop the scalar tile from RHS CB
    cb_post_rhs.pop_front(1);
}
