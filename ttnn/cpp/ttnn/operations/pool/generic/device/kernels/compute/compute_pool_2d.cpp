// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

#define DEBUG_PRINT 1

#if DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"
#endif

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t nsticks_per_core_by_nblocks = get_compile_time_arg_val(3);

    constexpr uint32_t in_data_cb = get_compile_time_arg_val(7);
    constexpr uint32_t in_idx_cb = get_compile_time_arg_val(9);
    constexpr uint32_t tile_data_cb = get_compile_time_arg_val(13);
    constexpr uint32_t tile_idx_cb = get_compile_time_arg_val(14);

    constexpr uint32_t topk_output_tiles = 1;
    constexpr uint32_t topk_cb_tile_idx = 0;
    constexpr uint32_t data_dst_idx = 0;
    constexpr uint32_t index_dst_idx = 2;

    unary_op_init_common(in_data_cb, tile_data_cb);

    for (uint32_t n = 0; n < nsticks_per_core_by_nblocks; ++n) {
        cb_wait_front(in_data_cb, 1);
        cb_wait_front(in_idx_cb, 1);

        if (n == 0) {
            UNPACK(DPRINT << "IN CB " << ENDL());
            UNPACK(tt::compute::common::print_full_tile(in_idx_cb, 0));
            UNPACK(DPRINT << ENDL());
        }
        tensix_sync();

        tilize_init(in_data_cb, topk_output_tiles, tile_data_cb);
        // reconfig_data_format_srca(tile_data_cb);
        // pack_reconfig_data_format(tile_data_cb);
        tilize_block(in_data_cb, topk_output_tiles, tile_data_cb, topk_cb_tile_idx, topk_cb_tile_idx);
        tilize_uninit_with_dt(in_data_cb, in_idx_cb, tile_idx_cb);

        tensix_sync();

        tilize_init_short_with_dt(in_data_cb, in_idx_cb, topk_output_tiles, tile_idx_cb);
        // reconfig_data_format_srca(tile_idx_cb);
        // pack_reconfig_data_format(tile_idx_cb);
        tilize_block(in_idx_cb, topk_output_tiles, tile_idx_cb, topk_cb_tile_idx, topk_cb_tile_idx);
        tilize_uninit_with_dt(in_idx_cb, in_data_cb, tile_data_cb);

        tensix_sync();
        if (n == 0) {
            PACK(DPRINT << "TILE CB " << ENDL());
            PACK(tt::compute::common::print_full_tile(tile_idx_cb, 0));
            PACK(DPRINT << ENDL());
        }

        cb_pop_front(in_data_cb, 1);
        cb_pop_front(in_idx_cb, 1);
    }
}

}  // namespace NAMESPACE
