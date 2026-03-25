// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(0);  // value1 (multiplier b)
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(1);  // value2 (addend c)
    const auto scalar1_f = reinterpret_cast<const float*>(&packed_scalar1);
    const auto scalar2_f = reinterpret_cast<const float*>(&packed_scalar2);
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    init_sfpu(cb_input, cb_output);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 0);  // a -> dst[0]

            fill_tile_init();
            fill_tile(1, *scalar1_f);  // b (scalar) -> dst[1]
            fill_tile(2, *scalar2_f);  // c (scalar) -> dst[2]

            SFPU_OP_CHAIN_0  // expands to mac_tile_init(); mac_tile<DataFormat>(0,1,2,0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_output);
            tile_regs_release();

            cb_pop_front(cb_input, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
