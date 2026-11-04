// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t start_id = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t REQUESTED_WRITE_SIZE = get_compile_time_arg_val(1);
    constexpr uint32_t BATCH = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    const uint32_t destination_page_size = dst_args.get_aligned_page_size();
    const auto d = TensorAccessor(dst_args, dst_addr, destination_page_size);

    const uint32_t l1_page_stride = get_local_cb_interface(cb_out).fifo_page_size << cb_addr_shift;
    const uint32_t write_size_dst =
        REQUESTED_WRITE_SIZE < destination_page_size ? REQUESTED_WRITE_SIZE : destination_page_size;
    const uint32_t write_size = write_size_dst < l1_page_stride ? write_size_dst : l1_page_stride;

    Noc noc;
    CircularBuffer cb(cb_out);

    uint32_t tile_id = start_id;

    if constexpr (BATCH > 1) {
        // Each batch is read from the FIFO read pointer without wrapping, so the host admits batching only when a
        // core's num_tiles fit cb_out from its base: no batch ever crosses the end of the FIFO.
        uint32_t tiles_left = num_tiles;

        uint32_t batch = (tiles_left < BATCH) ? tiles_left : BATCH;
        cb.wait_front(batch);
        uint32_t l1_read_offset = 0;
        for (uint32_t t = 0; t < batch; t++) {
            noc.async_write(
                cb, d, write_size, {.offset_bytes = l1_read_offset}, {.page_id = tile_id++, .offset_bytes = 0});
            l1_read_offset += l1_page_stride;
        }
        tiles_left -= batch;
        uint32_t prev_batch = batch;

        while (tiles_left > 0) {
            batch = (tiles_left < BATCH) ? tiles_left : BATCH;
            cb.wait_front(prev_batch + batch);
            noc.async_writes_flushed();
            cb.pop_front(prev_batch);

            l1_read_offset = 0;
            for (uint32_t t = 0; t < batch; t++) {
                noc.async_write(
                    cb, d, write_size, {.offset_bytes = l1_read_offset}, {.page_id = tile_id++, .offset_bytes = 0});
                l1_read_offset += l1_page_stride;
            }
            tiles_left -= batch;
            prev_batch = batch;
        }

        noc.async_writes_flushed();
        cb.pop_front(prev_batch);
    } else {
        for (uint32_t i = 0; i < num_tiles; i++) {
            cb.wait_front(1);
            noc.async_write(cb, d, write_size, {.offset_bytes = 0}, {.page_id = tile_id++, .offset_bytes = 0});
            noc.async_writes_flushed();
            cb.pop_front(1);
        }
    }
    noc.async_write_barrier();
}
