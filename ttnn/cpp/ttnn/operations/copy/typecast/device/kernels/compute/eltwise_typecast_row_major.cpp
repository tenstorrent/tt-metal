// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/typecast.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t input_cb = get_compile_time_arg_val(1);
    uint32_t output_cb = get_compile_time_arg_val(2);

    init_sfpu(input_cb, output_cb);

    // Process row-major pages (each page is a row) one by one.
    // We reuse tile-level APIs (copy_tile / TYPECAST_LLK / pack_tile) on each page,
    // assuming the page size/layout is compatible with the logical tile size and alignment
    // expected by these kernels.
    for (uint32_t page_index = 0; page_index < per_core_block_cnt; page_index++) {
        cb_reserve_back(output_cb, 1);
        tile_regs_acquire();
        cb_wait_front(input_cb, 1);

        // Copy a row-major page using the tile-move API from the input CB.
        copy_tile(input_cb, 0, 0);

        typecast_tile_init();

        // Apply typecast to the row-major page via tile compute API.
        TYPECAST_LLK(0);
        tile_regs_commit();
        tile_regs_wait();

        // Pack the row-major result page back to the output CB using the tile pack API.
        pack_tile(0, output_cb);

        cb_pop_front(input_cb, 1);
        tile_regs_release();
        cb_push_back(output_cb, 1);
    }
}
}  // namespace NAMESPACE
