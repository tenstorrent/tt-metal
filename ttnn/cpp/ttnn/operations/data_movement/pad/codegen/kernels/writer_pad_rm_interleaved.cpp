// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pad writer: RM interleaved, batched sequential stick writes.
// BRISC. Reuses the pipelined writer pattern from tt-transpose.
// Writes stick_size_out bytes per output stick, advancing by
// stick_size_out_aligned in the CB (L1-aligned stride).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    // Compile-time args
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t stick_size_out = get_compile_time_arg_val(1);
    constexpr uint32_t stick_size_out_aligned = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();
    constexpr uint32_t BATCH = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());

    // No explicit page-size override: the 2-arg TensorAccessor derives the
    // tensor's real bank-page pitch from its spec, so output pages address
    // correctly for any width/buffer type (a hand-computed pitch mis-addresses
    // every page >= 1 when it disagrees with the real pitch). Each write moves
    // only stick_size_out logical bytes.
    const auto d = TensorAccessor(dst_args, dst_addr);

    uint32_t stick_id = start_id;

    if constexpr (BATCH > 1) {
        // Pipelined batched writer: overlap NOC DMA with reader producing next batch
        uint32_t sticks_left = num_sticks;

        // Prime: issue first batch
        uint32_t batch = (sticks_left < BATCH) ? sticks_left : BATCH;
        cb_wait_front(cb_out, batch);
        uint32_t l1_addr = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < batch; t++) {
            noc_async_write_page(stick_id++, d, l1_addr, stick_size_out);
            l1_addr += stick_size_out_aligned;
        }
        sticks_left -= batch;
        uint32_t prev_batch = batch;

        // Steady state
        while (sticks_left > 0) {
            batch = (sticks_left < BATCH) ? sticks_left : BATCH;
            cb_wait_front(cb_out, prev_batch + batch);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, prev_batch);

            l1_addr = get_read_ptr(cb_out);
            for (uint32_t t = 0; t < batch; t++) {
                noc_async_write_page(stick_id++, d, l1_addr, stick_size_out);
                l1_addr += stick_size_out_aligned;
            }
            sticks_left -= batch;
            prev_batch = batch;
        }

        // Drain final batch
        noc_async_writes_flushed();
        cb_pop_front(cb_out, prev_batch);
    } else {
        for (uint32_t i = 0; i < num_sticks; i++) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(stick_id++, d, get_read_ptr(cb_out), stick_size_out);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
}
