// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file layernorm_compute_utils.h
 * @brief Utility functions for the layernorm compute kernels (tilize/untilize).
 */

#pragma once

#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"

#ifdef TILIZE_IN
#include "api/compute/tilize.h"
#endif  // TILIZE_IN

#ifdef UNTILIZE_OUT
#include "api/compute/pack_untilize.h"
#endif  // UNTILIZE_OUT

#ifdef TILIZE_IN
/*
 * Read 1 row-major block from cb_in_rm, tilize it and write to cb_in
 */
template <typename Block>
ALWI void tilize_row_major_block(
    const uint32_t cb_in_rm, const uint32_t cb_in, const uint32_t block_size, const Block& block) {
    reconfig_data_format(cb_in_rm, cb_in_rm);
    pack_reconfig_data_format(cb_in);

    tilize_init(cb_in_rm, block_size, cb_in);
    cb_wait_front(cb_in_rm, block.full_block_size());
    cb_reserve_back(cb_in, block.full_block_size());

    tilize_block(cb_in_rm, block.full_block_size(), cb_in);
    cb_push_back(cb_in, block.full_block_size());
    cb_pop_front(cb_in_rm, block.full_block_size());

    tilize_uninit(cb_in_rm, cb_in);
}

/*
 * Tilize all blocks of ROW_MAJOR input from cb_in_rm into cb_in (batch mode).
 * tilize_init and tilize_uninit wrap the full block loop; tilize_block is called per block.
 * Caller must reinitialize binary op hardware with binary_op_init_common after this returns,
 * since tilize_uninit reconfigures the hardware state.
 */
template <uint32_t block_size>
ALWI void tilize_all_blocks_to_cb(const uint32_t cb_in_rm, const uint32_t cb_in, const uint32_t Wt) {
    reconfig_data_format(cb_in_rm, cb_in_rm);
    pack_reconfig_data_format(cb_in);
    tilize_init(cb_in_rm, block_size, cb_in);
    for (auto block : norm::kernel_util::generic::blocks(Wt, block_size)) {
        cb_wait_front(cb_in_rm, block.full_block_size());
        cb_reserve_back(cb_in, block.full_block_size());
        tilize_block(cb_in_rm, block.full_block_size(), cb_in);
        cb_push_back(cb_in, block.full_block_size());
        cb_pop_front(cb_in_rm, block.full_block_size());
    }
    tilize_uninit(cb_in_rm, cb_in);
}
#endif

#ifdef UNTILIZE_OUT

/*
 * Read 1 tiled block from cb_out, pack it and write to cb_out_rm as row-major block
 */
template <typename Block, uint32_t block_size>
ALWI void untilize_row_major_block(const uint32_t cb_out, const uint32_t cb_out_rm, const Block& block) {
    reconfig_data_format(cb_out, cb_out);  // Handle fp32_dest_acc_en=True cases

    pack_untilize_init<block_size, block_size>(cb_out, cb_out_rm);
    cb_wait_front(cb_out, block.full_block_size());
    cb_reserve_back(cb_out_rm, block.full_block_size());
    pack_untilize_block<block_size, block_size>(cb_out, 1, cb_out_rm);
    cb_push_back(cb_out_rm, block.full_block_size());
    cb_pop_front(cb_out, block.full_block_size());
    pack_untilize_uninit(cb_out_rm);
}

/*
 * Pack-untilize all blocks from cb_out into cb_out_rm as row-major.
 */
template <uint32_t block_size>
ALWI void untilize_all_blocks_from_cb(const uint32_t cb_out, const uint32_t cb_out_rm, const uint32_t Wt) {
    // If fp32_dest_acc_en=True and dtype == bfloat16, then intermediate cb were set to float32 while output is set to
    // bfloat16. To prevent data corruption, we reconfig data format
    reconfig_data_format(cb_out, cb_out);
    pack_untilize_init<block_size, block_size>(cb_out, cb_out_rm);
    for (auto block : norm::kernel_util::generic::blocks(Wt, block_size)) {
        cb_wait_front(cb_out, block.full_block_size());
        cb_reserve_back(cb_out_rm, block.full_block_size());
        pack_untilize_block<block_size, block_size>(cb_out, 1, cb_out_rm);
        cb_push_back(cb_out_rm, block.full_block_size());
        cb_pop_front(cb_out, block.full_block_size());
    }
    pack_untilize_uninit(cb_out_rm);
}
#endif
