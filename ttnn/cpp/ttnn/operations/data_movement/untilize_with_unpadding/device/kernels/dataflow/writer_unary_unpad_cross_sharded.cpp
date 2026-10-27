// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "experimental/kernel_args.h"

// Writes a WIDTH_SHARDED/BLOCK_SHARDED input's untilized rows to a *differently-shard-typed*
// WIDTH_SHARDED/BLOCK_SHARDED output (e.g. WIDTH -> BLOCK), unlike
// writer_unary_unpad_batch_rows_sharded.cpp which assumes the output shard for this core lives
// on the same physical core as the input shard (a same-core L1-to-L1 copy). Here the executing
// core (owner of the input's column shard) may differ from the physically-owning core of the
// output shard, so the destination is addressed via TensorAccessor + noc_async_write_sharded,
// which already derives the row->page multiplication (pages-per-row) from the *output* tensor's
// own shard geometry - this kernel only needs to supply the logical row id and a byte offset for
// the column shard (col_shard_id * writer_page_size), not a pre-multiplied page id.
void kernel_main() {
    auto num_padded_tiles_per_batch = get_arg(args::num_padded_tiles_per_batch);
    auto num_unpadded_rows_per_batch = get_arg(args::num_unpadded_rows_per_batch);
    auto padded_block_row_size_bytes = get_arg(args::padded_block_row_size_bytes);
    auto unpadded_block_row_size_bytes = get_arg(args::unpadded_block_row_size_bytes);
    auto batch = get_arg(args::batch);
    auto col_byte_offset = get_arg(args::col_byte_offset);  // this core's output column-shard byte offset
    auto row_start_id = get_arg(args::row_start_id);        // this core's first absolute output row id

    // The destination's per-shard page size comes from the binding, which supplies the tensor's
    // aligned page size; noc_async_write_sharded derives the row->page split from it.
    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    // The untilized block the compute kernel packs; drained here row by row.
    DataflowBuffer dfb_untilize_out(dfb::untilize_out);

    uint32_t out_row_id = row_start_id;
    for (uint32_t b = 0; b < batch; ++b) {
        dfb_untilize_out.wait_front(num_padded_tiles_per_batch);
        uint32_t src_addr = dfb_untilize_out.get_read_ptr();

        for (uint32_t row = 0; row < num_unpadded_rows_per_batch; ++row) {
            tt::data_movement::common::noc_async_write_sharded(
                noc,
                src_addr,
                s,
                out_row_id,
                /*offset=*/col_byte_offset,
                /*size=*/unpadded_block_row_size_bytes);
            src_addr += padded_block_row_size_bytes;
            out_row_id += 1;
        }
        noc.async_write_barrier();
        dfb_untilize_out.pop_front(num_padded_tiles_per_batch);
    }
}
