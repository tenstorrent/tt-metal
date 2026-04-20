// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/fill.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    const float scalar1 = get_arg_val<float>(0);  // value1 (multiplier b)
    const float scalar2 = get_arg_val<float>(1);  // value2 (addend c)
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    experimental::CircularBuffer cb_input_obj(cb_input);
    experimental::CircularBuffer cb_output_obj(cb_output);

    init_sfpu(cb_input, cb_output);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_output_obj.reserve_back(per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_input_obj.wait_front(1);
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 0);  // a -> dst[0]

            fill_tile_init();
            fill_tile(1, scalar1);  // b (scalar) -> dst[1]
            fill_tile(2, scalar2);  // c (scalar) -> dst[2]

            SFPU_OP_CHAIN_0  // expands to mac_tile_init(); mac_tile<DataFormat>(0,1,2,0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_output);
            tile_regs_release();

            cb_input_obj.pop_front(1);
        }
        cb_output_obj.push_back(per_core_block_dim);
    }
}
