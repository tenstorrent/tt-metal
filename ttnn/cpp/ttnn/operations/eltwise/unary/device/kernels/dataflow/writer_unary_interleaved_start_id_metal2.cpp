// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_unary_interleaved_start_id.cpp.
// The legacy copy stays in place for its ~45 unmigrated co-borrowers; this fork carries the
// CB->DFB / named-binding rewrite for Metal 2.0 factories. Keep the two in sync until the last
// legacy consumer ports (then delete the legacy copy). See the metal2 port recipe's
// "Modifying a shared dataflow kernel" caution.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_pages = get_arg(args::num_pages);
    const auto start_id = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);

    // Get page size from the DFB (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = dfb_out.get_entry_size();

#ifdef OUT_SHARDED
    dfb_out.wait_front(num_pages);
#else

    // single-page ublocks (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(tensor::output);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        dfb_out.wait_front(onepage);
        noc.async_write(dfb_out, s, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        dfb_out.pop_front(onepage);
    }
    noc.async_write_barrier();
#endif
}
