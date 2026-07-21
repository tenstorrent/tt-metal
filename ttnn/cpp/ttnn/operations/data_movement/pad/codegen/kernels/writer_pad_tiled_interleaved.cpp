// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pad writer: TILE interleaved, batched sequential tile writes.
// BRISC. Same pipelined pattern as RM writer but with tiles.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    // Compile-time args
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();
    constexpr uint32_t BATCH = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());

    // Keep the three byte quantities independent. ``tile_bytes`` is the
    // requested tile payload supplied by the host, TensorAccessorArgs owns the
    // destination's physical page pitch, and the CB descriptor owns the L1
    // staging stride. Keep them independent for placement-specific alignment
    // and nonstandard page configurations.
    const uint32_t destination_page_size = dst_args.get_aligned_page_size();
    const uint32_t l1_page_stride = get_local_cb_interface(cb_out).fifo_page_size << cb_addr_shift;
    const uint32_t write_size_dst = tile_bytes < destination_page_size ? tile_bytes : destination_page_size;
    const uint32_t write_size = write_size_dst < l1_page_stride ? write_size_dst : l1_page_stride;
    const auto d = TensorAccessor(dst_args, dst_addr, destination_page_size);

    uint32_t tile_id = start_id;

    if constexpr (BATCH > 1) {
        uint32_t tiles_left = num_tiles;

        // Prime: issue first batch
        uint32_t batch = (tiles_left < BATCH) ? tiles_left : BATCH;
        cb_wait_front(cb_out, batch);
        uint32_t l1_addr = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < batch; t++) {
            noc_async_write_page(tile_id++, d, l1_addr, write_size);
            l1_addr += l1_page_stride;
        }
        tiles_left -= batch;
        uint32_t prev_batch = batch;

        // Steady state
        while (tiles_left > 0) {
            batch = (tiles_left < BATCH) ? tiles_left : BATCH;
            cb_wait_front(cb_out, prev_batch + batch);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, prev_batch);

            l1_addr = get_read_ptr(cb_out);
            for (uint32_t t = 0; t < batch; t++) {
                noc_async_write_page(tile_id++, d, l1_addr, write_size);
                l1_addr += l1_page_stride;
            }
            tiles_left -= batch;
            prev_batch = batch;
        }

        // Drain final batch
        noc_async_writes_flushed();
        cb_pop_front(cb_out, prev_batch);
    } else {
        for (uint32_t i = 0; i < num_tiles; i++) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(tile_id++, d, get_read_ptr(cb_out), write_size);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
}
