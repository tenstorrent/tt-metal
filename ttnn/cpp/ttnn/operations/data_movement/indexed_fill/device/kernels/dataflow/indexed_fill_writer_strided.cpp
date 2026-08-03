// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Writer for indexed_fill generic path (interleaved output, arbitrary dim).
//
// Pops pages from the data DFB and writes them to scattered output page IDs using a
// slices × outer × inner loop:
//
//   output_page_id = outer * outer_stride + my_slice * inner_count + inner
//
// For dim=0 with one slice per core: outer_count=1, matching original sequential behavior.
void kernel_main() {
    const uint32_t page_size = get_arg(args::page_size);
    const uint32_t outer_count = get_arg(args::outer_count);
    const uint32_t inner_count = get_arg(args::inner_count);
    const uint32_t outer_stride = get_arg(args::outer_stride);
    const uint32_t slice_start = get_arg(args::slice_start);
    const uint32_t num_slices = get_arg(args::num_slices);

    if (num_slices == 0) {
        return;
    }

    const auto dst = TensorAccessor(tensor::output);

    Noc noc;
    DataflowBuffer dfb(dfb::in0);

    for (uint32_t s = 0; s < num_slices; ++s) {
        const uint32_t my_slice = slice_start + s;
        for (uint32_t outer = 0; outer < outer_count; ++outer) {
            for (uint32_t inner = 0; inner < inner_count; ++inner) {
                const uint32_t pid = outer * outer_stride + my_slice * inner_count + inner;
                dfb.wait_front(1);
                noc.async_write(dfb, dst, page_size, {.offset_bytes = 0}, {.page_id = pid});
                noc.async_writes_flushed();
                dfb.pop_front(1);
            }
        }
    }
    noc.async_write_barrier();
}
