// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    constexpr uint32_t N = get_named_compile_time_arg_val("N");
    constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
    constexpr uint32_t num_rows = get_named_compile_time_arg_val("num_rows");
    constexpr auto src_args = TensorAccessorArgs<0>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t end_row = get_arg_val<uint32_t>(2);

    const auto s0 = TensorAccessor(src_args, src_addr);
    experimental::CircularBuffer cb(tt::CBIndex::c_0);
    experimental::Noc noc;

    uint32_t curr_addr = src_addr;
    for (uint32_t row = start_row; row < end_row; ++row) {
        cb.reserve_back(1);
        noc.async_read(s0, cb, page_size, {.page_id = row}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb.push_back(1);
    }
}
