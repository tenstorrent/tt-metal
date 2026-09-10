// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Matmul's private copy of writer_unary_interleaved_start_id.cpp, converted to Metal 2.0.
//
// The filename is shared by ~20 unrelated copies across the tree. The nearest relatives are functionally
// identical to this one but each uses its own accessor name, so the names are not interchangeable
// when moving code between them:
//   - eltwise/unary/.../writer_unary_interleaved_start_id{,_metal2}.cpp
//   - experimental/quasar/matmul/.../writer_unary_interleaved_start_id.cpp
//
// See issue https://github.com/tenstorrent/tt-metal/issues/52228

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);
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
        noc.async_write(dfb_out, s, page_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_writes_flushed();
        dfb_out.pop_front(onepage);
    }
    noc.async_write_barrier();
#endif
}
