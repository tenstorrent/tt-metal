// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for ttnn.get_rows (interleaved row-major gather). The host hands each core a contiguous
// slice of the index order; for every output row this kernel decodes the source row index from the
// [N] index vector (pulled from DRAM in alignment-sized blocks into a private scratchpad) and
// NoC-reads the selected row of the [H, W] row-major table into the staging buffer, in order. The
// writer kernel drains the staged rows into the output tensor, preserving the index order.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t start_row = get_arg(args::start_row);
    uint32_t num_rows = get_arg(args::num_rows);
    uint32_t num_index_blocks = get_arg(args::num_index_blocks);
    uint32_t last_block_bytes = get_arg(args::last_block_bytes);

    constexpr uint32_t block_bytes = get_arg(args::block_bytes);
    constexpr uint32_t indices_per_block = get_arg(args::indices_per_block);
    constexpr uint32_t row_bytes = get_arg(args::row_bytes);

    const auto indices = TensorAccessor(tensor::indices);
    const auto input = TensorAccessor(tensor::input);
    Noc noc;

    // dfb_in0 stages the rows this core gathers; the writer kernel drains it to the output tensor.
    DataflowBuffer dfb_in0(dfb::in0);
    // The index scratch is this reader's private page for the block of indices it is working
    // through: the NoC reads into it via the scratchpad object and the RISC reads indices back.
    Scratchpad<volatile uint32_t> index_scratch(scratch::idx);

    uint32_t loaded_block = num_index_blocks;  // force the first block load
    for (uint32_t i = 0; i < num_rows; ++i) {
        const uint32_t global_row = start_row + i;
        const uint32_t block = global_row / indices_per_block;
        const uint32_t slot = global_row % indices_per_block;
        if (block != loaded_block) {
            const uint32_t bytes = (block == num_index_blocks - 1) ? last_block_bytes : block_bytes;
            noc.async_read(indices, index_scratch, bytes, {.page_id = 0, .offset_bytes = block * block_bytes}, {});
            noc.async_read_barrier();
            loaded_block = block;
        }
        uint32_t src_row;
        {
            // The NoC rewrites this scratchpad with each new index block, so on Quasar the DM core's
            // L2 may still hold the previous block; scoped_lock invalidates the line before the read.
            // No-op on WH/BH.
            auto index_lock = index_scratch.scoped_lock(slot, 1);
            src_row = index_scratch[slot];
        }

        dfb_in0.reserve_back(1);
        const CoreLocalMem<uint8_t> dst(dfb_in0.get_write_ptr());
        noc.async_read(input, dst, row_bytes, {.page_id = src_row, .offset_bytes = 0}, {});
        noc.async_read_barrier();
        dfb_in0.push_back(1);
    }
}
