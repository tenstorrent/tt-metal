// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This is the Metal 2.0 fork of reader_unary_interleaved_start_id.cpp, which lives beside it.
// Ops ported to Metal 2.0 bind this file; the original serves the consumers still on the legacy API.
// Until the last of them migrates and the original is retired, changes here likely belong there too.
//
// The binding names below (dfb::in, tensor::src) and the named argument set are this fork's
// interface: every later consumer inherits them, so they are taken from the kernel's own vocabulary
// rather than any one op's locals, and are not renamed once a consumer exists.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#ifdef IMPLICIT_SYNC
#include "api/kernel_thread_globals.h"
#endif

void kernel_main() {
    const auto num_pages = get_arg(args::num_pages);
    const auto start_id = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer dfb(dfb::in);

    // Get page size from the DFB (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = dfb.get_entry_size();

    // ublocks size defined in pages (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(tensor::src);

#ifdef IMPLICIT_SYNC
    // Thread t of N reads pages start_id + t, start_id + t + N, ...: each TXN_ID read fills the next
    // entry of the strided DFB and posts its credit when it lands.
    for (uint32_t i = start_id + get_my_thread_id(); i < start_id + num_pages; i += get_num_threads()) {
        noc.async_read<NocOptions::TXN_ID>(s, dfb, {.page_id = i}, {});
    }
#else
// read a ublock of pages from src to CB, and then push the ublock to unpacker
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        dfb.reserve_back(onepage);
        noc.async_read(s, dfb, page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb.push_back(onepage);
    }
#endif
}
