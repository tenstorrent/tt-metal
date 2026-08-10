// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pad writer: RM interleaved, batched sequential stick writes.
// BRISC. Reuses the pipelined writer pattern from tt-transpose.
// Writes stick_size_out bytes per output stick, advancing by
// stick_size_out_aligned in the CB (L1-aligned stride).
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "rm_shard_split.h"

void kernel_main() {
    // Runtime args
    uint32_t dst_addr        = get_arg_val<uint32_t>(0);
    uint32_t num_sticks      = get_arg_val<uint32_t>(1);
    uint32_t start_id        = get_arg_val<uint32_t>(2);

    // Compile-time args
    constexpr uint32_t cb_out               = get_compile_time_arg_val(0);
    constexpr uint32_t stick_size_out       = get_compile_time_arg_val(1);
    constexpr uint32_t stick_size_out_aligned = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();
    constexpr uint32_t BATCH = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());
    // ROW_MAJOR width/block-sharded destinations: a page is one shard-width
    // slice, not a row, so `page_id = stick_id` names the wrong bytes. The host
    // emits (0, 0) for every other destination and the header then collapses to
    // the historical single write. Located via next_compile_time_args_offset()
    // so a sharded destination's longer accessor arg block shifts them
    // transparently.
    constexpr uint32_t DST_PAGES_PER_ROW =
        get_compile_time_arg_val(dst_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t DST_LOGICAL_PAGE_SIZE =
        get_compile_time_arg_val(dst_args.next_compile_time_args_offset() + 2);

    // No explicit page-size override: the 2-arg TensorAccessor derives the
    // tensor's real bank-page pitch from its spec, so output pages address
    // correctly for any width/buffer type (a hand-computed pitch mis-addresses
    // every page >= 1 when it disagrees with the real pitch). Each write moves
    // only stick_size_out logical bytes.
    const auto d = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    CircularBuffer out_cb(cb_out);

    uint32_t stick_id = start_id;

    if constexpr (BATCH > 1) {
        // Pipelined batched writer: overlap NOC DMA with reader producing next batch
        uint32_t sticks_left = num_sticks;

        // Prime: issue first batch
        uint32_t batch = (sticks_left < BATCH) ? sticks_left : BATCH;
        out_cb.wait_front(batch);
        uint32_t l1_offset = 0;
        for (uint32_t t = 0; t < batch; t++) {
            ttdm::noc_write_row_split<DST_PAGES_PER_ROW, DST_LOGICAL_PAGE_SIZE>(
                noc, out_cb, l1_offset, d, stick_id++, /*dst_offset=*/0,
                stick_size_out);
            l1_offset += stick_size_out_aligned;
        }
        sticks_left -= batch;
        uint32_t prev_batch = batch;

        // Steady state
        while (sticks_left > 0) {
            batch = (sticks_left < BATCH) ? sticks_left : BATCH;
            out_cb.wait_front(prev_batch + batch);
            noc.async_writes_flushed();
            out_cb.pop_front(prev_batch);

            l1_offset = 0;
            for (uint32_t t = 0; t < batch; t++) {
                ttdm::noc_write_row_split<DST_PAGES_PER_ROW, DST_LOGICAL_PAGE_SIZE>(
                    noc, out_cb, l1_offset, d, stick_id++, /*dst_offset=*/0,
                    stick_size_out);
                l1_offset += stick_size_out_aligned;
            }
            sticks_left -= batch;
            prev_batch = batch;
        }

        // Drain final batch
        noc.async_writes_flushed();
        out_cb.pop_front(prev_batch);
    } else {
        for (uint32_t i = 0; i < num_sticks; i++) {
            out_cb.wait_front(1);
            ttdm::noc_write_row_split<DST_PAGES_PER_ROW, DST_LOGICAL_PAGE_SIZE>(
                noc, out_cb, /*src_offset=*/0, d, stick_id++, /*dst_offset=*/0,
                stick_size_out);
            noc.async_writes_flushed();
            out_cb.pop_front(1);
        }
    }
    noc.async_write_barrier();
}
