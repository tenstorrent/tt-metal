// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_u32 = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(cb_in0, cb_out);
    binop_with_scalar_tile_init();

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        cb_wait_front(cb_in0, num_tiles_per_cycle);
        cb_wait_front(cb_in1, num_tiles_per_cycle);
        cb_wait_front(cb_in2, num_tiles_per_cycle);

        cb_reserve_back(cb_out, num_tiles_per_cycle);

        tile_regs_acquire();

        // Copy inputs to DST registers (as done by ternary reader)
        copy_tile_to_dst_init_short(cb_in0);
        copy_tile(cb_in0, 0, 0);  // input_a -> DST[0]

        copy_tile_to_dst_init_short(cb_in1);
        copy_tile(cb_in1, 0, 1);  // input_b -> DST[1]

        copy_tile_to_dst_init_short(cb_in2);
        copy_tile(cb_in2, 0, 2);  // input_c -> DST[2]

        // Step 1: input_b * input_c -> DST[3]
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 3);  // DST[1] * DST[2] -> DST[3]

        // Step 2: (input_b * input_c) * value -> DST[3]
        // Skip multiplication if value == 1.0 to avoid precision loss
        if (scalar_u32 != 0x3f800000) {     // 1.0f in IEEE 754
            mul_unary_tile(3, scalar_u32);  // DST[3] * scalar -> DST[3]
        }

        // Step 3: input_a + (input_b * input_c * value) -> DST[0]
        add_binary_tile_init();
        add_binary_tile(0, 3, 0);  // DST[0] + DST[3] -> DST[0]

        tile_regs_commit();
        tile_regs_wait();

        // Pack the result from DST[0] to output
        pack_tile(0, cb_out);

        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_in0, num_tiles_per_cycle);
        cb_pop_front(cb_in1, num_tiles_per_cycle);
        cb_pop_front(cb_in2, num_tiles_per_cycle);
    }
}
}  // namespace NAMESPACE
