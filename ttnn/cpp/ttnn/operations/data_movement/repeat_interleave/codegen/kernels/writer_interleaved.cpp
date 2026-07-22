// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sequential page writer for interleaved tensors (TILE and RM).
// Supports optional batching via BATCH compile-time arg.
// When BATCH > 1: pipelined — overlaps NOC DMA of batch N with compute
// delivering batch N+1. Requires cb_out depth >= 2 * BATCH.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t REQUESTED_WRITE_SIZE = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();
    constexpr uint32_t BATCH = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());

    // TensorAccessorArgs owns the destination's physical address pitch. The
    // caller's historical CT[1] is only the requested bytes copied per page;
    // it may be a compact payload or a placement-specific aligned size.
    const uint32_t destination_page_size = dst_args.get_aligned_page_size();
    const auto d = TensorAccessor(dst_args, dst_addr, destination_page_size);

    // Source-CB stride and destination transport pitch are independent. Most
    // paths configure the CB at the destination page size, but RM concat can
    // read a 16-byte BF16 stick from a 32-byte-aligned DRAM source and write it
    // to a 16-byte interleaved-L1 page.  Its CB must retain the larger 32-byte
    // source page; deriving the read stride from destination pitch then
    // consumes source padding as every other output stick.  The CB descriptor is
    // the authority for its L1 page stride. Convert the CB field to bytes
    // explicitly; cb_addr_shift is zero on dataflow RISCs but
    // keeping it here prevents this generic writer from depending on that fact.
    const uint32_t l1_page_stride = get_local_cb_interface(cb_out).fifo_page_size << cb_addr_shift;
    // Never read beyond the staging slot or write beyond the destination page.
    // The minimum preserves placement-specific or nonstandard page layouts and
    // is unchanged for ordinary equal-pitch cases.
    const uint32_t write_size_dst =
        REQUESTED_WRITE_SIZE < destination_page_size ? REQUESTED_WRITE_SIZE : destination_page_size;
    const uint32_t write_size = write_size_dst < l1_page_stride ? write_size_dst : l1_page_stride;

    uint32_t tile_id = start_id;

    if constexpr (BATCH > 1) {
        // Pipelined batched writer: overlap NOC DMA of batch N with compute
        // delivering batch N+1. While we wait for the new batch to arrive
        // (cb_wait_front), the NOC finishes reading the previous batch from L1,
        // so the subsequent flush is nearly free.
        uint32_t tiles_left = num_tiles;

        // Prime the pipeline: issue first batch without prior flush
        uint32_t batch = (tiles_left < BATCH) ? tiles_left : BATCH;
        cb_wait_front(cb_out, batch);
        uint32_t l1_addr = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < batch; t++) {
            noc_async_write_page(tile_id++, d, l1_addr, write_size);
            l1_addr += l1_page_stride;
        }
        tiles_left -= batch;
        uint32_t prev_batch = batch;

        // Steady state: wait for old + new tiles, then flush/pop old, issue new.
        // We must wait for prev_batch + batch because prev_batch tiles haven't
        // been popped yet and are still counted as "available" by cb_wait_front.
        while (tiles_left > 0) {
            batch = (tiles_left < BATCH) ? tiles_left : BATCH;
            cb_wait_front(cb_out, prev_batch + batch);  // wait for NEW batch to arrive
            noc_async_writes_flushed();                 // flush prev (NOC drained during wait)
            cb_pop_front(cb_out, prev_batch);           // reclaim prev batch space

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
