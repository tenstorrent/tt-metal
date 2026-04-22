// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for RM→RM typecast with optional page padding.
//
// Writes only `actual_page_bytes` (true row width) to DRAM, discarding any
// padding bytes that the compute kernel added to reach padded_page_bytes.

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t actual_page_bytes = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();

    const auto s = TensorAccessor(dst_args, dst_addr);
    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id_out);

    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb.wait_front(1);
        noc.async_write(cb, s, actual_page_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        cb.pop_front(1);
    }
}
