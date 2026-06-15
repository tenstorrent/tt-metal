// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file layernorm_compute_utils.h
 * @brief Utility functions for the layernorm compute kernels (tilize/untilize).
 */

#pragma once

#include "api/dataflow/circular_buffer.h"
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
    CircularBuffer& cb_in_rm, CircularBuffer& cb_in, const uint32_t block_size, const Block& block) {
    reconfig_data_format(cb_in_rm.get_cb_id(), cb_in_rm.get_cb_id());
    pack_reconfig_data_format(cb_in.get_cb_id());

    tilize_init(cb_in_rm.get_cb_id(), block_size, cb_in.get_cb_id());
    cb_in_rm.wait_front(block.full_block_size());
    cb_in.reserve_back(block.full_block_size());

    tilize_block(cb_in_rm.get_cb_id(), block.full_block_size(), cb_in.get_cb_id());
    cb_in.push_back(block.full_block_size());
    cb_in_rm.pop_front(block.full_block_size());

    tilize_uninit(cb_in_rm.get_cb_id(), cb_in.get_cb_id());
}

/*
 * Tilize all blocks of ROW_MAJOR input from cb_in_rm into cb_in (batch mode).
 * tilize_init and tilize_uninit wrap the full block loop; tilize_block is called per block.
 * Caller must reinitialize binary op hardware with binary_op_init_common after this returns,
 * since tilize_uninit reconfigures the hardware state.
 */
template <uint32_t block_size>
ALWI void tilize_all_blocks_to_cb(CircularBuffer& cb_in_rm, CircularBuffer& cb_in, const uint32_t Wt) {
    reconfig_data_format(cb_in_rm.get_cb_id(), cb_in_rm.get_cb_id());
    pack_reconfig_data_format(cb_in.get_cb_id());
    tilize_init(cb_in_rm.get_cb_id(), block_size, cb_in.get_cb_id());
    for (auto block : norm::kernel_util::generic::blocks(Wt, block_size)) {
        cb_in_rm.wait_front(block.full_block_size());
        cb_in.reserve_back(block.full_block_size());
        tilize_block(cb_in_rm.get_cb_id(), block.full_block_size(), cb_in.get_cb_id());
        cb_in.push_back(block.full_block_size());
        cb_in_rm.pop_front(block.full_block_size());
    }
    tilize_uninit(cb_in_rm.get_cb_id(), cb_in.get_cb_id());
}
#endif

#ifdef UNTILIZE_OUT

/*
 * Read 1 tiled block from cb_out, pack it and write to cb_out_rm as row-major block
 */
template <typename Block, uint32_t block_size>
ALWI void untilize_row_major_block(CircularBuffer& cb_out, CircularBuffer& cb_out_rm, const Block& block) {
    reconfig_data_format(cb_out.get_cb_id(), cb_out.get_cb_id());  // Handle fp32_dest_acc_en=True cases

    pack_untilize_init<block_size, block_size>(cb_out.get_cb_id(), cb_out_rm.get_cb_id());
    cb_out.wait_front(block.full_block_size());
    cb_out_rm.reserve_back(block.full_block_size());
    pack_untilize_block<block_size, block_size>(cb_out.get_cb_id(), 1, cb_out_rm.get_cb_id());
    cb_out_rm.push_back(block.full_block_size());
    cb_out.pop_front(block.full_block_size());
    pack_untilize_uninit(cb_out_rm.get_cb_id());
}

/*
 * Pack-untilize all blocks from cb_out into cb_out_rm as row-major.
 */
template <uint32_t block_size>
ALWI void untilize_all_blocks_from_cb(CircularBuffer& cb_out, CircularBuffer& cb_out_rm, const uint32_t Wt) {
    // If fp32_dest_acc_en=True and dtype == bfloat16, then intermediate cb were set to float32 while output is set to
    // bfloat16. To prevent data corruption, we reconfig data format
    reconfig_data_format(cb_out.get_cb_id(), cb_out.get_cb_id());
    pack_untilize_init<block_size, block_size>(cb_out.get_cb_id(), cb_out_rm.get_cb_id());
    for (auto block : norm::kernel_util::generic::blocks(Wt, block_size)) {
        cb_out.wait_front(block.full_block_size());
        cb_out_rm.reserve_back(block.full_block_size());
        pack_untilize_block<block_size, block_size>(cb_out.get_cb_id(), 1, cb_out_rm.get_cb_id());
        cb_out_rm.push_back(block.full_block_size());
        cb_out.pop_front(block.full_block_size());
    }
    pack_untilize_uninit(cb_out_rm.get_cb_id());
}
#endif
