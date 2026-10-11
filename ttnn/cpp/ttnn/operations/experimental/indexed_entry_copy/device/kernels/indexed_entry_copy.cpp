// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Copies this core's page range of one entry between a multi-entry tensor and a single-entry tensor.
// The entry is resolved here from the device-resident index, so a captured trace follows whatever the
// caller writes into `index` before each replay.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t page_stride = get_arg(args::page_stride);
    constexpr uint32_t batch_pages = get_arg(args::batch_pages);
    constexpr uint32_t pages_per_entry = get_arg(args::pages_per_entry);
    constexpr uint32_t num_entries = get_arg(args::num_entries);
    constexpr uint32_t index_scale = get_arg(args::index_scale);
    constexpr uint32_t index_offset = get_arg(args::index_offset);
    constexpr uint32_t index_page_size = get_arg(args::index_page_size);
    constexpr bool indexed_is_source = get_arg(args::indexed_is_source);
    const uint32_t first_page = get_arg(args::first_page);
    const uint32_t num_pages = get_arg(args::num_pages);

    if (num_pages == 0) {
        return;
    }

    Noc noc;

    // Read the whole aligned page holding the scalar: a short or unaligned read of a DRAM scalar is not
    // guaranteed to land.
    DataflowBuffer index_staging(dfb::index_staging);
    const uint32_t index_l1 = index_staging.get_write_ptr();
    const auto index = TensorAccessor(tensor::index);
    noc.async_read(index, CoreLocalMem<uint32_t>(index_l1), index_page_size, {.page_id = 0}, {});
    noc.async_read_barrier();
    const uint32_t index_value = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_l1)[0];

    // index_scale >= 1 and index_offset >= 0, so an index at or past num_entries can only land past it; checking
    // first also keeps the product below from wrapping into range.
    if (index_value >= num_entries) {
        return;
    }
    const uint32_t entry = index_value * index_scale + index_offset;
    if (entry >= num_entries) {
        return;
    }
    const uint32_t entry_base = entry * pages_per_entry + first_page;
    const uint32_t source_base = indexed_is_source ? entry_base : first_page;
    const uint32_t destination_base = indexed_is_source ? first_page : entry_base;

    const auto source = TensorAccessor(tensor::source);
    const auto destination = TensorAccessor(tensor::destination);
    DataflowBuffer staging(dfb::staging);
    const uint32_t staging_l1 = staging.get_write_ptr();

    for (uint32_t done = 0; done < num_pages; done += batch_pages) {
        const uint32_t count = num_pages - done < batch_pages ? num_pages - done : batch_pages;
        for (uint32_t i = 0; i < count; ++i) {
            noc.async_read(
                source,
                CoreLocalMem<uint8_t>(staging_l1 + i * page_stride),
                page_size,
                {.page_id = source_base + done + i},
                {});
        }
        noc.async_read_barrier();
        for (uint32_t i = 0; i < count; ++i) {
            noc.async_write(
                CoreLocalMem<uint8_t>(staging_l1 + i * page_stride),
                destination,
                page_size,
                {},
                {.page_id = destination_base + done + i});
        }
        noc.async_write_barrier();
    }
}
