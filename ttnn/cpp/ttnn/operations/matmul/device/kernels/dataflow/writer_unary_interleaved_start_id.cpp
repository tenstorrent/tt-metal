// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Matmul's private copy of writer_unary_interleaved_start_id.cpp, converted in place to Metal 2.0.
// MatmulMultiCoreProgramFactory is its only binder.
//
// The filename is shared by ~20 unrelated copies across the tree, so "the" writer_unary kernel does
// not exist — every op that binds one binds a different file. The nearest relatives are functionally
// identical to this one but each picked its own accessor name, so the names are NOT interchangeable
// when moving code between them:
//   - eltwise/unary/.../writer_unary_interleaved_start_id{,_metal2}.cpp — the legacy original and
//     its Metal 2.0 fork, serving the eltwise consumers. The fork uses `tensor::dst`.
//   - experimental/quasar/matmul/.../writer_unary_interleaved_start_id.cpp — the quasar fork's copy,
//     already on Metal 2.0. Uses `tensor::out`, and `cb_out` for the DFB local. Changes here likely
//     belong there too.
// All three bind the DFB as `dfb::out`; only the tensor accessor name differs.
//
// TODO(#52228): retire this duplication. The issue records why it exists, the full consumer list,
// and the sunset plan: https://github.com/tenstorrent/tt-metal/issues/52228

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

    // Page size comes from the DFB entry size (works for both TILE and ROW_MAJOR layouts). Read it
    // from the DFB object: the legacy get_local_cb_interface(...).fifo_page_size is stale for
    // Metal 2.0 DFBs.
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
