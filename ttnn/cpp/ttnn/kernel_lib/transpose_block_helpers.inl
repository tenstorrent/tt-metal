// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/circular_buffer.h"

namespace compute_kernel_lib {

template <uint32_t in_block_num_tiles, uint32_t block_size>
FORCE_INLINE void transpose_tile_block(uint32_t in_transpose_cb_id, uint32_t in_cb_id) {
    constexpr uint32_t num_blocks = in_block_num_tiles / block_size;
    constexpr uint32_t last_block_size = in_block_num_tiles % block_size;

    ::experimental::CircularBuffer in_transpose_cb(in_transpose_cb_id);
    ::experimental::CircularBuffer in_cb(in_cb_id);

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        in_transpose_cb.wait_front(block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            transpose_wh_tile(in_transpose_cb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        in_transpose_cb.pop_front(block_size);

        in_cb.reserve_back(block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            pack_tile(tile_idx, in_cb_id);
        }
        tile_regs_release();
        in_cb.push_back(block_size);
    }

    if constexpr (last_block_size > 0) {
        in_transpose_cb.wait_front(last_block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            transpose_wh_tile(in_transpose_cb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        in_transpose_cb.pop_front(last_block_size);

        in_cb.reserve_back(last_block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            pack_tile(tile_idx, in_cb_id);
        }
        tile_regs_release();
        in_cb.push_back(last_block_size);
    }
}

template <
    uint32_t in0_block_num_tiles,
    uint32_t in0_transpose_cb_id,
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    bool in1_transpose_tile,
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t in0_block_w,
    uint32_t mm_partials_cb_id>
ALWI void TransposePreKBlock<
    in0_block_num_tiles,
    in0_transpose_cb_id,
    in0_cb_id,
    in1_cb_id,
    in1_transpose_tile,
    out_subblock_w,
    out_subblock_h,
    in0_block_w,
    mm_partials_cb_id>::operator()(uint32_t, uint32_t, bool) const {
    reconfig_data_format_srca(in1_cb_id, in0_transpose_cb_id);
    transpose_wh_init_short(in0_transpose_cb_id);
    PACK((pack_reconfig_data_format(in0_cb_id)));
#ifdef PACKER_L1_ACC
    PACK((llk_pack_reconfig_l1_acc(0)));
#endif
    transpose_tile_block<in0_block_num_tiles>(in0_transpose_cb_id, in0_cb_id);
    mm_block_init_short_with_dt(
        in0_cb_id,
        in1_cb_id,
        in0_transpose_cb_id,
        in1_transpose_tile,
        out_subblock_w,
        out_subblock_h,
        in0_block_w);
    PACK((pack_reconfig_data_format(mm_partials_cb_id)));
}

}  // namespace compute_kernel_lib
