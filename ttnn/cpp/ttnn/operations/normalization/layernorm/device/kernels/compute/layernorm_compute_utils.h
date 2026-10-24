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
    dfb_in_rm.wait_front(block.full_block_size());
    dfb_in.reserve_back(block.full_block_size());

    tilize_block(dfb_in_rm.get_id(), block.full_block_size(), dfb_in.get_id(), 0, 0, block_size);
    dfb_in.push_back(block.full_block_size());
    dfb_in_rm.pop_front(block.full_block_size());
}

/*
 * Tilize all blocks of ROW_MAJOR input from dfb_in_rm into dfb_in (batch mode).
 * Caller must reinitialize binary op hardware with binary_op_init_common after this returns,
 * since tilize_block leaves the hardware in its uninitialized state.
 */
template <uint32_t block_size>
ALWI void tilize_all_blocks_to_dfb(DataflowBuffer& dfb_in_rm, DataflowBuffer& dfb_in, const uint32_t Wt) {
    for (auto block : norm::kernel_util::generic::blocks(Wt, block_size)) {
        dfb_in_rm.wait_front(block.full_block_size());
        dfb_in.reserve_back(block.full_block_size());
        tilize_block(dfb_in_rm.get_id(), block.full_block_size(), dfb_in.get_id(), 0, 0, block_size);
        dfb_in.push_back(block.full_block_size());
        dfb_in_rm.pop_front(block.full_block_size());
    }
}
#endif

#ifdef UNTILIZE_OUT

/*
 * Read 1 tiled block from dfb_out, pack it and write to dfb_out_rm as row-major block
 */
template <typename Block, uint32_t block_size>
ALWI void untilize_row_major_block(DataflowBuffer& dfb_out, DataflowBuffer& dfb_out_rm, const Block& block) {
    dfb_out.wait_front(block.full_block_size());
    dfb_out_rm.reserve_back(block.full_block_size());
    pack_untilize_block<block_size, block_size>(dfb_out.get_id(), 1, dfb_out_rm.get_id());
    dfb_out_rm.push_back(block.full_block_size());
    dfb_out.pop_front(block.full_block_size());
}

/*
 * Pack-untilize all blocks from dfb_out into dfb_out_rm as row-major.
 */
template <uint32_t block_size>
ALWI void untilize_all_blocks_from_dfb(DataflowBuffer& dfb_out, DataflowBuffer& dfb_out_rm, const uint32_t Wt) {
    for (auto block : norm::kernel_util::generic::blocks(Wt, block_size)) {
        dfb_out.wait_front(block.full_block_size());
        dfb_out_rm.reserve_back(block.full_block_size());
        pack_untilize_block<block_size, block_size>(dfb_out.get_id(), 1, dfb_out_rm.get_id());
        dfb_out_rm.push_back(block.full_block_size());
        dfb_out.pop_front(block.full_block_size());
    }
}
#endif
