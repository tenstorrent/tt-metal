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

#define TILE_WIDTH 32
#define FACE_WIDTH 16
#define FACE_HEIGHT 16

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t in_data_cb = get_compile_time_arg_val(7);
    constexpr uint32_t in_idx_cb = get_compile_time_arg_val(9);
    constexpr uint32_t tile_tmp_cb_id = get_compile_time_arg_val(13);
    constexpr uint32_t tile_idx_tmp_cb_id = get_compile_time_arg_val(14);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(15);
    constexpr uint32_t out_idx_cb_id = get_compile_time_arg_val(16);
    constexpr bool return_indices = (bool)get_compile_time_arg_val(18);

    constexpr uint32_t topk_output_tiles = 1;
    constexpr uint32_t topk_cb_tile_idx = 0;
    constexpr uint32_t data_dst_idx = 0;
    constexpr uint32_t index_dst_idx = 2;
    constexpr uint32_t num_out_sticks = 1;
    const uint32_t output_faces = 2;

    cb_reserve_back(out_cb_id, output_faces);
    cb_reserve_back(out_idx_cb_id, output_faces);
    cb_wait_front(in_data_cb, 1);
    cb_wait_front(in_idx_cb, 1);

    tensix_sync();  // make sure tensix is idle for init
    unary_op_init_common(in_data_cb, tile_tmp_cb_id);
    tensix_sync();  // make sure tensix is idle for init

    cb_reserve_back(tile_tmp_cb_id, topk_output_tiles);

    tilize_init(in_data_cb, topk_output_tiles, tile_tmp_cb_id);
    tilize_block(in_data_cb, topk_output_tiles, tile_tmp_cb_id, topk_cb_tile_idx, topk_cb_tile_idx);
    tilize_uninit_with_dt(in_data_cb, in_idx_cb, tile_idx_tmp_cb_id);

    cb_push_back(tile_tmp_cb_id, topk_output_tiles);
    cb_wait_front(tile_tmp_cb_id, topk_output_tiles);

    cb_reserve_back(tile_idx_tmp_cb_id, topk_output_tiles);

    tilize_init_short_with_dt(in_data_cb, in_idx_cb, topk_output_tiles, tile_idx_tmp_cb_id);
    tilize_block(in_idx_cb, topk_output_tiles, tile_idx_tmp_cb_id, topk_cb_tile_idx, topk_cb_tile_idx);
    tilize_uninit(in_idx_cb, tile_idx_tmp_cb_id);

    cb_push_back(tile_idx_tmp_cb_id, topk_output_tiles);
    cb_wait_front(tile_idx_tmp_cb_id, topk_output_tiles);

    tile_regs_acquire();

    pack_reconfig_data_format(tile_tmp_cb_id);
    copy_tile_init(tile_tmp_cb_id);
    copy_tile(tile_tmp_cb_id, 0, data_dst_idx);

    copy_tile_to_dst_init_short_with_dt(tile_tmp_cb_id, tile_idx_tmp_cb_id);
    copy_tile(tile_idx_tmp_cb_id, 0, index_dst_idx);

    // sort tile 0 descending, phase 0 through 4 which is log2(32-1)
    topk_tile_init();
    ckernel::topk_local_sort(data_dst_idx, 0, 4, 0);

    // Pop the temporary circular buffers after processing
    cb_pop_front(tile_tmp_cb_id, topk_output_tiles);
    cb_pop_front(tile_idx_tmp_cb_id, topk_output_tiles);
    cb_pop_front(in_data_cb, 1);
    cb_pop_front(in_idx_cb, 1);

    dprint_tensix_dest_reg(data_dst_idx);
    // dprint_tensix_dest_reg(index_dst_idx);

    tile_regs_commit();
    tile_regs_wait();

    tensix_sync();  // make sure tensix is idle for init
    pack_untilize_dest_init<topk_output_tiles>(out_cb_id, num_out_sticks, output_faces);
    tensix_sync();  // make sure tensix is idle for init

    pack_untilize_dest<topk_output_tiles>(out_cb_id, 1, 0, num_out_sticks, output_faces, data_dst_idx);
    PACK(tt::compute::common::print_tile_rows(out_cb_id, 1));

    pack_reconfig_data_format(out_idx_cb_id);
    pack_untilize_dest<topk_output_tiles>(out_idx_cb_id, 1, 0, num_out_sticks, output_faces, index_dst_idx);
    // PACK(DPRINT << "OUT CB " << ENDL());
    // PACK(tt::compute::common::print_tile_rows(out_idx_cb_id, 1));

    pack_untilize_uninit(out_cb_id);

    cb_push_back(out_cb_id, output_faces);
    cb_push_back(out_idx_cb_id, output_faces);

    tile_regs_release();
}

}  // namespace NAMESPACE
