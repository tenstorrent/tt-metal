// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sequential stick writer for RM interleaved tensors with repeat.
//
// 64B page-alignment fix (matches ops/expand's proven RM path): writes xfer_size
// bytes per page, where xfer_size is the buffer's aligned page size
// (round_up(stick, dram_alignment) — 64B on Blackhole). Each staging-DFB slot is
// l1_stride (== aligned page) bytes, so transferring the whole aligned page never
// over-reads the adjacent slot, and every NOC transfer is 64B-aligned and a 64B
// multiple. The TensorAccessor (no explicit page_size) addresses page `id` at
// base + id*aligned_page_size, exactly where the host packs/unpacks
// buffer.page_size() real bytes on read-back; the copied per-page padding is
// trimmed by the host. Shared by both repeat RM builders (higher-dim and
// last-dim); both pass [real_stick, aligned_page] so this kernel uses the aligned
// page (xfer_size) for the transfer size AND the L1 stride.
//
// Named CT args: xfer_size, l1_stride, batch
// Bindings:      tensor::dst (output tensor), dfb::out (staged pages to drain)
// Named RT args: num_pages, start_id
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_pages = get_arg(args::num_pages);
    uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t xfer_size = get_arg(args::xfer_size);
    constexpr uint32_t l1_stride = get_arg(args::l1_stride);
    constexpr uint32_t BATCH = get_arg(args::batch);

    const auto d = TensorAccessor(tensor::dst);

    Noc noc;
    // dfb::out — the output pages staged by the reader, drained to the output tensor here.
    DataflowBuffer dfb(dfb::out);

    uint32_t page_id = start_id;

    if constexpr (BATCH > 1) {
        uint32_t pages_left = num_pages;

        // Prime the pipeline
        uint32_t batch = (pages_left < BATCH) ? pages_left : BATCH;
        dfb.wait_front(batch);
        uint32_t l1_offset = 0;
        for (uint32_t t = 0; t < batch; t++) {
            noc.async_write(dfb, d, xfer_size, {.offset_bytes = l1_offset}, {.page_id = page_id++, .offset_bytes = 0});
            l1_offset += l1_stride;
        }
        pages_left -= batch;
        uint32_t prev_batch = batch;

        // Steady state
        while (pages_left > 0) {
            batch = (pages_left < BATCH) ? pages_left : BATCH;
            dfb.wait_front(prev_batch + batch);
            noc.async_writes_flushed();
            dfb.pop_front(prev_batch);

            l1_offset = 0;
            for (uint32_t t = 0; t < batch; t++) {
                noc.async_write(
                    dfb, d, xfer_size, {.offset_bytes = l1_offset}, {.page_id = page_id++, .offset_bytes = 0});
                l1_offset += l1_stride;
            }
            pages_left -= batch;
            prev_batch = batch;
        }

        // Drain
        noc.async_writes_flushed();
        dfb.pop_front(prev_batch);
    } else {
        for (uint32_t i = 0; i < num_pages; i++) {
            dfb.wait_front(1);
            noc.async_write(dfb, d, xfer_size, {.offset_bytes = 0}, {.page_id = page_id++, .offset_bytes = 0});
            noc.async_writes_flushed();
            dfb.pop_front(1);
        }
    }
    noc.async_write_barrier();
}
