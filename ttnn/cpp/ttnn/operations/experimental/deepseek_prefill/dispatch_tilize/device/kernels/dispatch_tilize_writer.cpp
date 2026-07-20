// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

// Pipelined interleaved DRAM writer. The stock tilize writer flushes after every single tile
// (noc_async_writes_flushed), so it never has more than one write outstanding — on an
// interleaved TILE output that serialises against the 8 DRAM banks and caps write BW at ~70%
// of peak. Here we issue a whole block's tile-writes back-to-back and barrier once, so
// consecutive tiles (which land in different banks) drain concurrently.
//
// The output CB holds exactly one block (num_pages == tiles_per_block) and the compute pushes
// one block at a time, so wait_front(tiles_per_block) always leaves the read pointer at the CB
// base → the block's pages are contiguous and the per-tile source offset is valid.
void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;
    const uint32_t tiles_per_block = get_local_cb_interface(cb_id_out).fifo_num_pages;

    Noc noc;
    DataflowBuffer dfb(cb_id_out);

#ifdef OUT_SHARDED
    dfb.wait_front(num_pages);
#else
    const auto s = TensorAccessor(dst_args, dst_addr);

    uint32_t page = start_id;
    uint32_t remaining = num_pages;
    while (remaining > 0) {
        const uint32_t batch = (remaining < tiles_per_block) ? remaining : tiles_per_block;
        dfb.wait_front(batch);
        for (uint32_t k = 0; k < batch; ++k) {
            noc.async_write(dfb, s, page_bytes, {.offset_bytes = k * page_bytes}, {.page_id = page + k});
        }
        noc.async_write_barrier();
        dfb.pop_front(batch);
        page += batch;
        remaining -= batch;
    }
#endif
}
