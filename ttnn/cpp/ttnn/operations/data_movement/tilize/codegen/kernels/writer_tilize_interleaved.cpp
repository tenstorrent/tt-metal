// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tile writer for tilize: writes tilized output tiles to DRAM.
// Supports optional batching via BATCH compile-time arg for pipelining.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();
    constexpr uint32_t BATCH = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());

    const auto d = TensorAccessor(dst_args, dst_addr, page_size);

    Noc noc;
    CircularBuffer cb_out_buf(cb_out);

    uint32_t tile_id = start_id;

    if constexpr (BATCH > 1) {
        uint32_t tiles_left = num_tiles;

        // Prime the pipeline
        uint32_t batch = (tiles_left < BATCH) ? tiles_left : BATCH;
        cb_out_buf.wait_front(batch);
        uint32_t l1_offset = 0;
        for (uint32_t t = 0; t < batch; t++) {
            noc.async_write(
                cb_out_buf, d, page_size, {.offset_bytes = l1_offset}, {.page_id = tile_id++, .offset_bytes = 0});
            l1_offset += page_size;
        }
        tiles_left -= batch;
        uint32_t prev_batch = batch;

        // Steady state
        while (tiles_left > 0) {
            batch = (tiles_left < BATCH) ? tiles_left : BATCH;
            cb_out_buf.wait_front(prev_batch + batch);
            noc.async_writes_flushed();
            cb_out_buf.pop_front(prev_batch);

            // Offsets are relative to the CB read pointer, which pop_front
            // just advanced past the retired batch.
            l1_offset = 0;
            for (uint32_t t = 0; t < batch; t++) {
                noc.async_write(
                    cb_out_buf, d, page_size, {.offset_bytes = l1_offset}, {.page_id = tile_id++, .offset_bytes = 0});
                l1_offset += page_size;
            }
            tiles_left -= batch;
            prev_batch = batch;
        }

        noc.async_writes_flushed();
        cb_out_buf.pop_front(prev_batch);
    } else {
        for (uint32_t i = 0; i < num_tiles; i++) {
            cb_out_buf.wait_front(1);
            noc.async_write(cb_out_buf, d, page_size, {.offset_bytes = 0}, {.page_id = tile_id++, .offset_bytes = 0});
            noc.async_writes_flushed();
            cb_out_buf.pop_front(1);
        }
    }
    noc.async_write_barrier();
}
