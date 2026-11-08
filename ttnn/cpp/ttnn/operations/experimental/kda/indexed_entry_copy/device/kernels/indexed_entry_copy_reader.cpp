// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads this core's page range of one entry from the source into the staging DFB, a batch per push; when the
// `zero_fill_at_zero` word reads 0 it pushes zeroed batches instead and never reads the source.

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "indexed_entry.hpp"

void kernel_main() {
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t page_stride = get_arg(args::page_stride);
    constexpr uint32_t batch_pages = get_arg(args::batch_pages);
    constexpr bool indexed_is_source = get_arg(args::indexed_is_source);
    constexpr bool has_zero_fill_at_zero = get_arg(args::has_zero_fill_at_zero);
    const uint32_t first_page = get_arg(args::first_page);
    const uint32_t num_pages = get_arg(args::num_pages);

    const Noc noc;
    // The flag word is bound only with has_zero_fill_at_zero, hence get_token_if_present below.
    volatile uint32_t* fill_word = nullptr;
    if constexpr (has_zero_fill_at_zero) {
        const Scratchpad<volatile uint32_t> zero_fill_at_zero_word(
            *scratch::get_token_if_present<"zero_fill_at_zero_word">());
        fill_word = &zero_fill_at_zero_word[0];
        // Issued ahead of the index read, so resolve_entry_base's barrier lands both words.
        noc.async_read(
            TensorAccessor(*tensor::get_token_if_present<"zero_fill_at_zero">()),
            zero_fill_at_zero_word,
            sizeof(uint32_t),
            {.page_id = 0},
            {.offset_bytes = 0});
    }
    uint32_t entry_base;
    if (!resolve_entry_base(noc, first_page, entry_base)) {
        return;
    }
    const bool zero = has_zero_fill_at_zero && *fill_word == 0;
    const uint32_t source_base = indexed_is_source ? entry_base : first_page;

    const auto source = TensorAccessor(tensor::source);
    DataflowBuffer staging(dfb::staging);
    for (uint32_t done = 0; done < num_pages; done += batch_pages) {
        const uint32_t count = num_pages - done < batch_pages ? num_pages - done : batch_pages;
        staging.reserve_back(count);
        if (zero) {
            noc.async_write_zeros(staging, count * page_stride);
            noc.write_zeros_l1_barrier();
            staging.push_back(count);
            continue;
        }
        const uint32_t batch_addr = staging.get_write_ptr();
        for (uint32_t i = 0; i < count; ++i) {
            noc.async_read(
                source,
                CoreLocalMem<uint32_t>(batch_addr + i * page_stride),
                page_size,
                {.page_id = source_base + done + i},
                {});
        }
        noc.async_read_barrier();
        staging.push_back(count);
    }
}
