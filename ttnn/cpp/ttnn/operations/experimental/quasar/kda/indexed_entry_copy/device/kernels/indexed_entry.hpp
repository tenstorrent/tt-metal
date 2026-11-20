// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Resolves the entry from the device-resident index, so a captured trace follows whatever the caller writes into
// `index` before each replay. Sets `entry_base` to this core's first page in the multi-entry tensor, or returns false
// when the index selects no entry.
inline bool resolve_entry_base(const Noc& noc, uint32_t first_page, uint32_t& entry_base) {
    constexpr uint32_t pages_per_entry = get_arg(args::pages_per_entry);
    constexpr uint32_t index_limit = get_arg(args::index_limit);
    constexpr uint32_t index_scale = get_arg(args::index_scale);
    constexpr uint32_t index_offset = get_arg(args::index_offset);

    Scratchpad<volatile uint32_t> index_word(scratch::index_word);
    noc.async_read(TensorAccessor(tensor::index), index_word, sizeof(uint32_t), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const uint32_t index_value = index_word.local_mem()[0];
    // The host derived index_limit so that passing it means entry < num_entries without a wrap.
    if (index_value >= index_limit) {
        return false;
    }
    entry_base = (index_value * index_scale + index_offset) * pages_per_entry + first_page;
    return true;
}
