// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Copies this core's page range of one entry between a multi-entry tensor and a single-entry tensor.
// The entry is resolved here from the device-resident index, so a captured trace follows whatever the
// caller writes into `index` before each replay.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t page_stride = get_arg(args::page_stride);
    constexpr uint32_t batch_pages = get_arg(args::batch_pages);
    constexpr uint32_t pages_per_entry = get_arg(args::pages_per_entry);
    constexpr uint32_t index_limit = get_arg(args::index_limit);
    constexpr uint32_t index_scale = get_arg(args::index_scale);
    constexpr uint32_t index_offset = get_arg(args::index_offset);
    constexpr bool indexed_is_source = get_arg(args::indexed_is_source);
    const uint32_t first_page = get_arg(args::first_page);
    const uint32_t num_pages = get_arg(args::num_pages);

    const Noc read_noc(0);
    const Noc write_noc(1);
    Scratchpad<volatile uint32_t> staging(scratch::staging);

    read_noc.async_read(TensorAccessor(tensor::index), staging, sizeof(uint32_t), {.page_id = 0}, {.offset_bytes = 0});
    read_noc.async_read_barrier();
    const uint32_t index_value = staging.local_mem()[0];
    // The host derived index_limit so that passing it means entry < num_entries without a wrap.
    if (index_value >= index_limit) {
        return;
    }
    const uint32_t entry_base = (index_value * index_scale + index_offset) * pages_per_entry + first_page;
    const uint32_t source_base = indexed_is_source ? entry_base : first_page;
    const uint32_t destination_base = indexed_is_source ? first_page : entry_base;

    const auto source = TensorAccessor(tensor::source);
    const auto destination = TensorAccessor(tensor::destination);
    for (uint32_t done = 0; done < num_pages; done += batch_pages) {
        const uint32_t count = num_pages - done < batch_pages ? num_pages - done : batch_pages;
        for (uint32_t i = 0; i < count; ++i) {
            read_noc.async_read(
                source, staging, page_size, {.page_id = source_base + done + i}, {.offset_bytes = i * page_stride});
        }
        read_noc.async_read_barrier();
        for (uint32_t i = 0; i < count; ++i) {
            write_noc.async_write(
                staging,
                destination,
                page_size,
                {.offset_bytes = i * page_stride},
                {.page_id = destination_base + done + i});
        }
        write_noc.async_write_barrier();
    }
}
