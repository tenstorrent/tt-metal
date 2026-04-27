// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr auto cb_id_dst = tt::CBIndex::c_2;

    experimental::Noc noc;
    experimental::CircularBuffer cb_dst(cb_id_dst);

#if DST_SHARDED
    cb_dst.wait_front(num_pages);
#else
    constexpr uint32_t onepage = 1;
    constexpr auto dst_args = TensorAccessorArgs<0, 0>();
    const auto dst = TensorAccessor(dst_args, dst_addr);

    uint32_t end_id = start_id + num_pages;
#if RM_INTERLEAVED
    const uint32_t chunks_per_row = get_arg_val<uint32_t>(3);
    const uint32_t chunk_size = get_arg_val<uint32_t>(4);
    const uint32_t last_chunk_size = get_arg_val<uint32_t>(5);
    const uint32_t rows_per_tile = get_arg_val<uint32_t>(6);
    const uint32_t total_rows = get_arg_val<uint32_t>(7);

    for (uint32_t block = start_id; block < end_id; ++block) {
        uint32_t base_page = block * rows_per_tile;
        uint32_t remaining = total_rows - base_page;
        uint32_t actual_rows = (rows_per_tile < remaining) ? rows_per_tile : remaining;

        for (uint32_t j = 0; j < chunks_per_row; ++j) {
            uint32_t bytes = (j == chunks_per_row - 1) ? last_chunk_size : chunk_size;
            cb_dst.wait_front(onepage);
            for (uint32_t r = 0; r < actual_rows; ++r) {
                noc.async_write(
                    cb_dst,
                    dst,
                    bytes,
                    {.offset_bytes = r * bytes},
                    {.page_id = base_page + r, .offset_bytes = j * chunk_size});
            }
            noc.async_writes_flushed();
            cb_dst.pop_front(onepage);
        }
    }
    noc.async_write_barrier();
#else
    const uint32_t page_bytes = get_local_cb_interface(cb_id_dst).fifo_page_size;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_dst.wait_front(onepage);
        noc.async_write(cb_dst, dst, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        cb_dst.pop_front(onepage);
    }
    noc.async_write_barrier();
#endif
#endif
}
