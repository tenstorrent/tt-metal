// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_interleaved.cpp (same directory). Identical dataflow logic; only the
// resource plumbing moves to the Metal 2.0 named bindings: the output CB index becomes dfb::out, the
// destination tensor becomes tensor::dst (so the dst_addr runtime arg, the positional
// TensorAccessorArgs compile-time args and the accessor's explicit page-size argument all disappear),
// and the remaining positional arguments become named ones. Forked rather than converted in place
// because the legacy file is still bound by repeat_interleave/codegen on the positional-arg API;
// delete this fork once every binder has adopted the named-binding form.
//
// Sequential page writer for interleaved tensors (TILE and RM).
// Supports optional batching via the `batch` compile-time arg.
// When BATCH > 1: pipelined — overlaps NOC DMA of batch N with compute
// delivering batch N+1. Requires dfb::out depth >= 2 * BATCH.
//
// Named CT args: requested_write_size, batch
// Bindings:      tensor::dst (destination tensor), dfb::out (staged pages to drain)
// Named RT args: num_tiles, start_id
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t REQUESTED_WRITE_SIZE = get_arg(args::requested_write_size);
    constexpr uint32_t BATCH = get_arg(args::batch);

    // The destination binding owns the destination's physical address pitch. The
    // caller's requested_write_size is only the requested bytes copied per page;
    // it may be a compact payload or a placement-specific aligned size.
    const auto d = TensorAccessor(tensor::dst);
    const uint32_t destination_page_size = d.get_aligned_page_size();

    Noc noc;
    // dfb::out — the pages staged upstream, drained to the destination tensor here.
    DataflowBuffer dfb(dfb::out);

    // Source-DFB stride and destination transport pitch are independent. Most
    // paths configure the DFB at the destination page size, but RM concat can
    // read a 16-byte BF16 stick from a 32-byte-aligned DRAM source and write it
    // to a 16-byte interleaved-L1 page.  Its DFB must retain the larger 32-byte
    // source page; deriving the read stride from destination pitch then
    // consumes source padding as every other output stick.  The DFB spec is
    // the authority for its L1 page stride; get_entry_size() reports it in bytes,
    // so this generic writer never depends on the address-unit shift of the RISC
    // it runs on.
    const uint32_t l1_page_stride = dfb.get_entry_size();
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
        // (wait_front), the NOC finishes reading the previous batch from L1,
        // so the subsequent flush is nearly free.
        uint32_t tiles_left = num_tiles;

        // Prime the pipeline: issue first batch without prior flush
        uint32_t batch = (tiles_left < BATCH) ? tiles_left : BATCH;
        dfb.wait_front(batch);
        uint32_t l1_read_offset = 0;
        for (uint32_t t = 0; t < batch; t++) {
            noc.async_write(
                dfb, d, write_size, {.offset_bytes = l1_read_offset}, {.page_id = tile_id++, .offset_bytes = 0});
            l1_read_offset += l1_page_stride;
        }
        tiles_left -= batch;
        uint32_t prev_batch = batch;

        // Steady state: wait for old + new tiles, then flush/pop old, issue new.
        // We must wait for prev_batch + batch because prev_batch tiles haven't
        // been popped yet and are still counted as "available" by wait_front.
        while (tiles_left > 0) {
            batch = (tiles_left < BATCH) ? tiles_left : BATCH;
            dfb.wait_front(prev_batch + batch);  // wait for NEW batch to arrive
            noc.async_writes_flushed();          // flush prev (NOC drained during wait)
            dfb.pop_front(prev_batch);           // reclaim prev batch space

            l1_read_offset = 0;
            for (uint32_t t = 0; t < batch; t++) {
                noc.async_write(
                    dfb, d, write_size, {.offset_bytes = l1_read_offset}, {.page_id = tile_id++, .offset_bytes = 0});
                l1_read_offset += l1_page_stride;
            }
            tiles_left -= batch;
            prev_batch = batch;
        }

        // Drain final batch
        noc.async_writes_flushed();
        dfb.pop_front(prev_batch);
    } else {
        for (uint32_t i = 0; i < num_tiles; i++) {
            dfb.wait_front(1);
            noc.async_write(dfb, d, write_size, {.offset_bytes = 0}, {.page_id = tile_id++, .offset_bytes = 0});
            noc.async_writes_flushed();
            dfb.pop_front(1);
        }
    }
    noc.async_write_barrier();
}
