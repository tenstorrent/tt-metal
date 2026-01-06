// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c
    constexpr auto cb_out = tt::CBIndex::c_3;
    const bool scalar_is_not_1 = scalar_arg != 1u;

    unary_op_init_common(cb_in0, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // Wait for input_b and input_c first (needed for first computation)
        cb_wait_front(cb_in1, num_tiles_per_cycle);
        cb_wait_front(cb_in2, num_tiles_per_cycle);

        tile_regs_acquire();

        // Step 1: Load B and C, compute B * C -> DST[1]
        copy_tile_to_dst_init_short(cb_in1);
        copy_tile(cb_in1, 0, 0);  // input_b -> DST[0]

        copy_tile_to_dst_init_short(cb_in2);
        copy_tile(cb_in2, 0, 1);  // input_c -> DST[1]

        mul_binary_tile_init();
        mul_binary_tile(0, 1, 1);  // DST[0] * DST[1] -> DST[1]

        // Done with cb_in1 and cb_in2, pop them early for pipeline efficiency
        cb_pop_front(cb_in1, num_tiles_per_cycle);
        cb_pop_front(cb_in2, num_tiles_per_cycle);

        // Step 2: (input_b * input_c) * value -> DST[1]
        if (scalar_is_not_1) {
            binop_with_scalar_tile_init();
            mul_unary_tile(1, scalar_arg);  // DST[1] * scalar -> DST[1]
        }

        // Now wait for input_a (only when we need it)
        cb_wait_front(cb_in0, num_tiles_per_cycle);

        // Step 3: Load A and add with result
        copy_tile_to_dst_init_short(cb_in0);
        copy_tile(cb_in0, 0, 0);  // input_a -> DST[0]

        add_binary_tile_init();
        add_binary_tile(0, 1, 0);  // DST[0] + DST[1] -> DST[0]

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
