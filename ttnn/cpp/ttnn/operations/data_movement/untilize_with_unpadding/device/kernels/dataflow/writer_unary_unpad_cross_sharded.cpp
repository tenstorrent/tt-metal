// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

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
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_padded_tiles_per_batch = get_arg_val<uint32_t>(1);
    uint32_t num_unpadded_rows_per_batch = get_arg_val<uint32_t>(2);
    uint32_t padded_block_row_size_bytes = get_arg_val<uint32_t>(3);
    uint32_t unpadded_block_row_size_bytes = get_arg_val<uint32_t>(4);
    uint32_t batch = get_arg_val<uint32_t>(5);
    uint32_t col_byte_offset = get_arg_val<uint32_t>(6);  // this core's output column-shard byte offset
    uint32_t row_start_id = get_arg_val<uint32_t>(7);     // this core's first absolute output row id

    constexpr uint32_t writer_page_size = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    constexpr uint32_t dfb_id_untilize_out = 16;

    const auto s = TensorAccessor(dst_args, dst_addr, writer_page_size);

    Noc noc;
    DataflowBuffer dfb_untilize_out(dfb_id_untilize_out);

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
