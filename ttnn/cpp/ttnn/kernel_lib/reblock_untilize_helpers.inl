// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace compute_kernel_lib {

template <uint32_t out_subblock_w, uint32_t out_block_w>
inline void reblock_and_untilize(
    uint32_t num_subblocks_w,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h,
    uint32_t interm_cb_id,
    uint32_t out_cb_id) {
    const uint32_t num_tiles_in_row_of_subblocks = mulsi3(out_subblock_num_tiles, num_subblocks_w);
    cb_wait_front(interm_cb_id, num_tiles_in_row_of_subblocks);

    uint32_t within_block_index = 0;
    for (uint32_t h = 0; h < out_subblock_h; h++) {
        uint32_t block_offset = 0;
        cb_reserve_back(out_cb_id, out_block_w);
        for (uint32_t n = 0; n < num_subblocks_w; n++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < out_subblock_w; w++) {
                copy_tile(interm_cb_id, block_offset + within_block_index + w, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<out_subblock_w, out_block_w>(out_cb_id, 1, n);
            tile_regs_release();
            block_offset += out_subblock_num_tiles;
        }
        cb_push_back(out_cb_id, out_block_w);
        within_block_index += out_subblock_w;
    }
    cb_pop_front(interm_cb_id, num_tiles_in_row_of_subblocks);
}

}  // namespace compute_kernel_lib
