// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(1);
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(2);
    const auto true_value = reinterpret_cast<const float*>(&packed_scalar1);
    const auto false_value = reinterpret_cast<const float*>(&packed_scalar2);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    init_sfpu(cb_input, cb_output);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_input, 1);
        cb_reserve_back(cb_output, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_input);
        copy_tile(cb_input, 0, 0);

        fill_tile_init();
#if defined(INP_INT32) || defined(INP_UINT32)
        fill_tile_int<DataFormat::Int32>(1, packed_scalar1);
        fill_tile_int<DataFormat::Int32>(2, packed_scalar2);
#endif
#if defined(INP_FLOAT) || defined(INP_FLOAT32)
        fill_tile(1, *true_value);
        fill_tile(2, *false_value);
#endif
        SFPU_OP_CHAIN_0
        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_output);
        tile_regs_release();

        cb_pop_front(cb_input, 1);
        cb_push_back(cb_output, 1);
    }
}
