// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sequential stick writer for RM interleaved tensors with repeat.
// Uses get_noc_addr for aligned page addressing, writes stick_size bytes.
//
// CT args: cb_out, stick_size, aligned_page_size, l1_slot_stride,
//          TensorAccessorArgs(out_t), BATCH
// RT args: dst_addr, num_pages, start_id
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_pages = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t aligned_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t l1_slot_stride = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();
    constexpr uint32_t BATCH = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());

    const auto d = TensorAccessor(dst_args, dst_addr, aligned_page_size);

    Noc noc;
    CircularBuffer cb(cb_out);

    uint32_t page_id = start_id;

    if constexpr (BATCH > 1) {
        uint32_t pages_left = num_pages;

        // Prime the pipeline
        uint32_t batch = (pages_left < BATCH) ? pages_left : BATCH;
        cb.wait_front(batch);
        uint32_t l1_offset = 0;
        for (uint32_t t = 0; t < batch; t++) {
            noc.async_write(cb, d, stick_size, {.offset_bytes = l1_offset}, {.page_id = page_id++, .offset_bytes = 0});
            l1_offset += l1_slot_stride;
        }
        pages_left -= batch;
        uint32_t prev_batch = batch;

        // Steady state
        while (pages_left > 0) {
            batch = (pages_left < BATCH) ? pages_left : BATCH;
            cb.wait_front(prev_batch + batch);
            noc.async_write_barrier();
            cb.pop_front(prev_batch);

            l1_offset = 0;
            for (uint32_t t = 0; t < batch; t++) {
                noc.async_write(
                    cb, d, stick_size, {.offset_bytes = l1_offset}, {.page_id = page_id++, .offset_bytes = 0});
                l1_offset += l1_slot_stride;
            }
            pages_left -= batch;
            prev_batch = batch;
        }

        // Drain
        noc.async_write_barrier();
        cb.pop_front(prev_batch);
    } else {
        for (uint32_t i = 0; i < num_pages; i++) {
            cb.wait_front(1);
            noc.async_write(cb, d, stick_size, {.offset_bytes = 0}, {.page_id = page_id++, .offset_bytes = 0});
            noc.async_write_barrier();
            cb.pop_front(1);
        }
    }
    noc.async_write_barrier();
}
