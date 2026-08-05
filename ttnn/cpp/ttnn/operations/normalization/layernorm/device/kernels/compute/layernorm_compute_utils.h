// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file layernorm_compute_utils.h
 * @brief Utility functions for the layernorm compute kernels (tilize/untilize).
 */

#pragma once

#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"

#ifdef TILIZE_IN
#include "api/compute/tilize.h"
#endif  // TILIZE_IN

#ifdef UNTILIZE_OUT
#include "api/compute/pack_untilize.h"
#endif  // UNTILIZE_OUT

#ifdef TILIZE_IN
/*
 * Read 1 row-major block from dfb_in_rm, tilize it and write to dfb_in
 */
template <typename Block>
ALWI void tilize_row_major_block(
    DataflowBuffer& dfb_in_rm, DataflowBuffer& dfb_in, const uint32_t block_size, const Block& block) {
    reconfig_data_format(dfb_in_rm.get_id(), dfb_in_rm.get_id());
    pack_reconfig_data_format(dfb_in.get_id());

    tilize_init(dfb_in_rm.get_id(), block_size, dfb_in.get_id());
    dfb_in_rm.wait_front(block.full_block_size());
    dfb_in.reserve_back(block.full_block_size());

    tilize_block(dfb_in_rm.get_id(), block.full_block_size(), dfb_in.get_id());
    dfb_in.push_back(block.full_block_size());
    dfb_in_rm.pop_front(block.full_block_size());

    tilize_uninit(dfb_in_rm.get_id(), dfb_in.get_id());
}

/*
 * Tilize all blocks of ROW_MAJOR input from dfb_in_rm into dfb_in (batch mode).
 * tilize_init and tilize_uninit wrap the full block loop; tilize_block is called per block.
 * Caller must reinitialize binary op hardware with binary_op_init_common after this returns,
 * since tilize_uninit reconfigures the hardware state.
 */
template <uint32_t block_size>
ALWI void tilize_all_blocks_to_cb(DataflowBuffer& dfb_in_rm, DataflowBuffer& dfb_in, const uint32_t Wt) {
    reconfig_data_format(dfb_in_rm.get_id(), dfb_in_rm.get_id());
    pack_reconfig_data_format(dfb_in.get_id());
    tilize_init(dfb_in_rm.get_id(), block_size, dfb_in.get_id());
    for (auto block : norm::kernel_util::generic::blocks(Wt, block_size)) {
        dfb_in_rm.wait_front(block.full_block_size());
        dfb_in.reserve_back(block.full_block_size());
        tilize_block(dfb_in_rm.get_id(), block.full_block_size(), dfb_in.get_id());
        dfb_in.push_back(block.full_block_size());
        dfb_in_rm.pop_front(block.full_block_size());
    }
    tilize_uninit(dfb_in_rm.get_id(), dfb_in.get_id());
}
#endif

#ifdef UNTILIZE_OUT

/*
 * Read 1 tiled block from dfb_out, pack it and write to dfb_out_rm as row-major block
 */
template <typename Block, uint32_t block_size>
ALWI void untilize_row_major_block(DataflowBuffer& dfb_out, DataflowBuffer& dfb_out_rm, const Block& block) {
    reconfig_data_format(dfb_out.get_id(), dfb_out.get_id());  // Handle fp32_dest_acc_en=True cases

    pack_untilize_init<block_size, block_size>(dfb_out.get_id(), dfb_out_rm.get_id());
    dfb_out.wait_front(block.full_block_size());
    dfb_out_rm.reserve_back(block.full_block_size());
    pack_untilize_block<block_size, block_size>(dfb_out.get_id(), 1, dfb_out_rm.get_id());
    dfb_out_rm.push_back(block.full_block_size());
    dfb_out.pop_front(block.full_block_size());
    pack_untilize_uninit(dfb_out_rm.get_id());
}

/*
 * Pack-untilize all blocks from dfb_out into dfb_out_rm as row-major.
 */
template <uint32_t block_size>
ALWI void untilize_all_blocks_from_cb(DataflowBuffer& dfb_out, DataflowBuffer& dfb_out_rm, const uint32_t Wt) {
    // If fp32_dest_acc_en=True and dtype == bfloat16, then intermediate cb were set to float32 while output is set to
    // bfloat16. To prevent data corruption, we reconfig data format
    reconfig_data_format(dfb_out.get_id(), dfb_out.get_id());
    pack_untilize_init<block_size, block_size>(dfb_out.get_id(), dfb_out_rm.get_id());
    for (auto block : norm::kernel_util::generic::blocks(Wt, block_size)) {
        dfb_out.wait_front(block.full_block_size());
        dfb_out_rm.reserve_back(block.full_block_size());
        pack_untilize_block<block_size, block_size>(dfb_out.get_id(), 1, dfb_out_rm.get_id());
        dfb_out_rm.push_back(block.full_block_size());
        dfb_out.pop_front(block.full_block_size());
    }
    pack_untilize_uninit(dfb_out_rm.get_id());
}
#endif
