// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/addcmul.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1
    const bool scalar_is_not_1 = scalar_arg != 1u;

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c
    constexpr auto cb_out = tt::CBIndex::c_3;

    // output = input_a + value * input_b * input_c
    binary_op_init_common(cb_in1, cb_in2, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // Wait for input_b and input_c first (needed for first computation)
        cb_wait_front(cb_in1, num_tiles_per_cycle);
        cb_wait_front(cb_in2, num_tiles_per_cycle);

        tile_regs_acquire();

        // (input_b * input_c)
        mul_tiles_init(cb_in1, cb_in2);
        mul_tiles(cb_in1, cb_in2, 0, 0, 0);

        // Done with cb_in1 and cb_in2, pop them early for pipeline efficiency
        cb_pop_front(cb_in1, num_tiles_per_cycle);
        cb_pop_front(cb_in2, num_tiles_per_cycle);

        // Step 2: (input_b * input_c) * value -> DST[0]
        if (scalar_is_not_1) {
            binop_with_scalar_tile_init();
            mul_unary_tile(0, scalar_arg);  // DST[0] * scalar -> DST[0]
        }

        // Now wait for input_a (only when we need it)
        cb_wait_front(cb_in0, num_tiles_per_cycle);

        // Step 3: Load A and add with result DST[0] + cb_in0 -> DST[0]
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in0);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in0, 0, 0);

        tile_regs_commit();
        tile_regs_wait();

        // Reserve output buffer only when ready to write
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        // Pack the result from DST[0] to output
        pack_tile(0, cb_out);

        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_in0, num_tiles_per_cycle);
    }
}
}  // namespace NAMESPACE
