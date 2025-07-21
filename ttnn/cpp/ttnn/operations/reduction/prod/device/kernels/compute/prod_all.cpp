// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "tt_metal/include/compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "tt_metal/include/compute_kernel_api.h"
#include "tt_metal/hw/inc/debug/dprint_pages.h"

namespace NAMESPACE {
void MAIN {
    const tt::CBIndex final_output_cb = tt::CBIndex::c_3;
    const tt::CBIndex input_cb = tt::CBIndex::c_0;
    const tt::CBIndex partial_prod_cb = tt::CBIndex::c_2;

    const int one_tile = 1;
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    binary_op_init_common(input_cb, partial_prod_cb, final_output_cb);

    reconfig_data_format(input_cb, partial_prod_cb);
    pack_reconfig_data_format(final_output_cb);

    // Ultimate goal is to have a single tile in the final output cb
    cb_reserve_back(final_output_cb, one_tile);

    // Copy the first tile to DST[0]
    cb_wait_front(input_cb, one_tile);
    tile_regs_acquire();

    copy_tile_to_dst_init_short(input_cb);
    copy_tile(input_cb, 0, 0);  // copy from c_in[0] to DST[0]

    cb_pop_front(input_cb, one_tile);
    mul_tiles_init(input_cb, partial_prod_cb);

    // When we have more than one tile, we can do the tile-wise multiplication of them all to yield one final tile
    for (uint32_t t = 1; t < num_tiles; t++) {
        // Save the current partial prod into CB
        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, partial_prod_cb);

        cb_reserve_back(partial_prod_cb, one_tile);
        cb_push_back(partial_prod_cb, one_tile);
        tile_regs_release();

        // Load the next input tile and multiply it with the partial prod
        cb_wait_front(input_cb, one_tile);

        tile_regs_acquire();

        mul_tiles(input_cb, partial_prod_cb, /*tile0=*/0, /*tile1=*/0, /*dst_tile=*/0);

        cb_pop_front(input_cb, one_tile);
        cb_pop_front(partial_prod_cb, one_tile);
    }

    tile_regs_commit();
    tile_regs_wait();

    pack_tile(0, final_output_cb);
    cb_push_back(final_output_cb, one_tile);
    tile_regs_release();
}
}  // namespace NAMESPACE
