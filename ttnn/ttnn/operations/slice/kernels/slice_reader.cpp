// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// slice reader (load_block): relays the source page of each of this cluster's output pages into the
// `slice_pages` ring. No compute helper applies (pure data movement); the implicit-sync NoC overload is the
// one-call-per-page primitive: a read tagged NocOptions::TXN_ID lands at the ring's cursor and the DM0
// interrupt handler posts one block (the ring's txn batch) at a time, so there is no reserve / barrier / push
// here. The block size lives only on the host (the ring's num_entries).
//
// Dealing: hart p of R handles output pages p, p+R, ... of [first_out_page, first_out_page + num_pages).
// Page map: output page m (row-major over out_page_shape[4]) is input page
//   src_base + sum_d m[d] * in_page_stride[d],
// advanced as an odometer (one divmod walk at the start, then R steps with carry per page).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t first_out_page = get_arg(args::first_out_page);
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t src_base = get_arg(args::src_base);
    const auto out_page_shape = get_arg(args::out_page_shape);
    const auto in_page_stride = get_arg(args::in_page_stride);
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t RANK = 4;

    DataflowBuffer cb_slice_pages(dfb::slice_pages);
    Noc noc;
    const auto input_accessor = TensorAccessor(tensor::input);

    const uint32_t thread = get_my_thread_id();
    if (thread < num_pages) {
        // Multi-index of this hart's first output page, and its source page.
        uint32_t idx[RANK];
        uint32_t rem = first_out_page + thread;
        for (int d = RANK - 1; d >= 0; --d) {
            idx[d] = rem % out_page_shape[d];
            rem /= out_page_shape[d];
        }
        uint32_t src_page = src_base;
        for (uint32_t d = 0; d < RANK; ++d) {
            src_page += idx[d] * in_page_stride[d];
        }

        for (uint32_t i = thread; i < num_pages; i += R) {
            noc.async_read<NocOptions::TXN_ID>(input_accessor, cb_slice_pages, {.page_id = src_page}, {});
            // Advance the odometer by R output pages. Unsigned wrap in the intermediate sums cancels out.
            idx[RANK - 1] += R;
            src_page += R * in_page_stride[RANK - 1];
            for (uint32_t d = RANK - 1; d > 0; --d) {
                while (idx[d] >= out_page_shape[d]) {
                    idx[d] -= out_page_shape[d];
                    src_page -= out_page_shape[d] * in_page_stride[d];
                    idx[d - 1] += 1;
                    src_page += in_page_stride[d - 1];
                }
            }
        }
    }
    // No barrier: firmware credits the last partial batch and drains the ring when the kernel returns.
    // Hand the hart's sticky tag register back with the user id: a tt-metal program that runs next with
    // untagged transfers would otherwise have them counted for a ring.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
