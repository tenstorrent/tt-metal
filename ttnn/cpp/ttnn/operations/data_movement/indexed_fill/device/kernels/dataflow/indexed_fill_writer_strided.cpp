// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Writer for indexed_fill generic path (interleaved output, arbitrary dim).
//
// Pops pages from the data CB and writes them to scattered output page IDs using a
// slices × outer × inner loop:
//
//   output_page_id = outer * outer_stride + my_slice * inner_count + inner
//
// For dim=0 with one slice per core: outer_count=1, matching original sequential behavior.
void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t page_size = get_arg_val<uint32_t>(1);
    const uint32_t outer_count = get_arg_val<uint32_t>(2);
    const uint32_t inner_count = get_arg_val<uint32_t>(3);
    const uint32_t outer_stride = get_arg_val<uint32_t>(4);
    const uint32_t slice_start = get_arg_val<uint32_t>(5);
    const uint32_t num_slices = get_arg_val<uint32_t>(6);

    if (num_slices == 0) {
        return;
    }

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    const auto dst = TensorAccessor(dst_args, output_addr);

    Noc noc;
    CircularBuffer cb(cb_id);

    for (uint32_t s = 0; s < num_slices; ++s) {
        const uint32_t my_slice = slice_start + s;
        for (uint32_t outer = 0; outer < outer_count; ++outer) {
            for (uint32_t inner = 0; inner < inner_count; ++inner) {
                const uint32_t pid = outer * outer_stride + my_slice * inner_count + inner;
                cb.wait_front(1);
                noc.async_write(cb, dst, page_size, {.offset_bytes = 0}, {.page_id = pid});
                noc.async_writes_flushed();
                cb.pop_front(1);
            }
        }
    }
    noc.async_write_barrier();
}
