// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gathered-prefix untilize compute (untilize with gathered_prefix_tensor): compute_untilize.cpp's pack_untilize per
// tile row, for the number of tile rows the reader derived from the device-held extent (word 0 of the mailbox CB).
//
// CT args: [Wt, cb_in, cb_out, max_bct, cb_meta]
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/untilize.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/cb_api.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t compute_num_blocks_per_column(uint32_t per_core_block_tile_cnt, uint32_t max_bct) {
    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (per_core_block_tile_cnt % bct == 0) {
            return per_core_block_tile_cnt / bct;
        }
    }
    return 1;
}

void kernel_main() {
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(0);  // Wt = tiles per tile-row
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t max_bct = get_compile_time_arg_val(3);
    constexpr uint32_t meta_cb_id = get_compile_time_arg_val(4);

    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_column(per_core_block_tile_cnt, max_bct);
    constexpr uint32_t block_ct_dim = per_core_block_tile_cnt / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = per_core_block_tile_cnt;

    CircularBuffer src_cb(src_cb_id);
    CircularBuffer out_cb(out_cb_id);
    CircularBuffer meta_cb(meta_cb_id);

    meta_cb.wait_front(1);
    const uint32_t per_core_block_cnt = ckernel::read_tile_value(meta_cb_id, 0, 0);  // tile rows of this core
    meta_cb.pop_front(1);

    compute_kernel_hw_startup(src_cb_id, out_cb_id);
    pack_untilize_init<block_ct_dim, full_ct_dim>(src_cb_id, out_cb_id);

    for (uint32_t r = 0; r < per_core_block_cnt; ++r) {
        out_cb.reserve_back(full_ct_dim);
        for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
            src_cb.wait_front(block_ct_dim);
            pack_untilize_block<block_ct_dim, full_ct_dim>(src_cb_id, 1, out_cb_id, b);
            src_cb.pop_front(block_ct_dim);
        }
        out_cb.push_back(full_ct_dim);
    }
    pack_untilize_uninit(out_cb_id);
}
