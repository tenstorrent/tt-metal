// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sequential stick writer for RM interleaved tensors with repeat.
//
// 64B page-alignment fix (matches ops/expand's proven RM path): writes xfer_size
// bytes per page, where xfer_size is the buffer's aligned page size
// (round_up(stick, dram_alignment) — 64B on Blackhole). Each CB slot is l1_stride
// (== aligned page) bytes, so transferring the whole aligned page never
// over-reads the adjacent slot, and every NOC transfer is 64B-aligned and a 64B
// multiple. The TensorAccessor (no explicit page_size) addresses page `id` at
// base + id*aligned_page_size, exactly where the host packs/unpacks
// buffer.page_size() real bytes on read-back; the copied per-page padding is
// trimmed by the host. Shared by both repeat RM builders (higher-dim and
// last-dim); both pass [real_stick, aligned_page] so this kernel uses the aligned
// page (CT slot 1) for the transfer size AND the L1 stride.
//
// CT args: cb_out, xfer_size, l1_stride, TensorAccessorArgs(out_t), BATCH
// RT args: dst_addr, num_pages, start_id
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_pages = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t xfer_size = get_compile_time_arg_val(1);
    constexpr uint32_t l1_stride = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();
    constexpr uint32_t BATCH = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());

    const auto d = TensorAccessor(dst_args, dst_addr);

    uint32_t page_id = start_id;

    if constexpr (BATCH > 1) {
        uint32_t pages_left = num_pages;

        // Prime the pipeline
        uint32_t batch = (pages_left < BATCH) ? pages_left : BATCH;
        cb_wait_front(cb_out, batch);
        uint32_t l1_addr = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < batch; t++) {
            noc_async_write_page(page_id++, d, l1_addr, xfer_size);
            l1_addr += l1_stride;
        }
        pages_left -= batch;
        uint32_t prev_batch = batch;

        // Steady state
        while (pages_left > 0) {
            batch = (pages_left < BATCH) ? pages_left : BATCH;
            cb_wait_front(cb_out, prev_batch + batch);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, prev_batch);

            l1_addr = get_read_ptr(cb_out);
            for (uint32_t t = 0; t < batch; t++) {
                noc_async_write_page(page_id++, d, l1_addr, xfer_size);
                l1_addr += l1_stride;
            }
            pages_left -= batch;
            prev_batch = batch;
        }

        // Drain
        noc_async_writes_flushed();
        cb_pop_front(cb_out, prev_batch);
    } else {
        for (uint32_t i = 0; i < num_pages; i++) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(page_id++, d, get_read_ptr(cb_out), xfer_size);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
}
