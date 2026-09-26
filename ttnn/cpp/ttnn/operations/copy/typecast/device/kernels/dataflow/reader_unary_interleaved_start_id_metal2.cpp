// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp.
// Identical dataflow logic; only the resource plumbing moves to the Metal 2.0 named bindings: the
// input CB index becomes dfb::in, the source tensor becomes tensor::input (so the src_addr runtime
// arg and the TensorAccessorArgs compile-time args both disappear), and the page count / start page
// become named runtime args. Forked rather than converted in place because the legacy file is
// instantiated by ~70 factories that are all still on the legacy positional-arg API; delete this
// fork once the eltwise/unary family adopts the same rewrite.
// Unlike the donor, one page can take several tile-sized entries, and only page_size bytes are copied.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t entries_per_page = get_arg(args::entries_per_page);
    constexpr uint32_t page_bytes = get_arg(args::page_size);

    const auto s = TensorAccessor(tensor::input);

    Noc noc;
    // dfb::in — the input pages this reader fills for the consumer downstream
    DataflowBuffer dfb(dfb::in);

// read a ublock of pages from src to CB, and then push the ublock to unpacker
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        dfb.reserve_back(entries_per_page);
        noc.async_read(s, dfb, page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb.push_back(entries_per_page);
    }
}
