// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t N = get_named_compile_time_arg_val("N");
    constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
    constexpr uint32_t num_rows = get_named_compile_time_arg_val("num_rows");
    constexpr auto src_args = TensorAccessorArgs<0>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t end_row = get_arg_val<uint32_t>(2);

    const auto s0 = TensorAccessor(src_args, src_addr);
    CircularBuffer cb(tt::CBIndex::c_0);
    Noc noc;

    for (uint32_t row = start_row; row < end_row; ++row) {
        cb.reserve_back(1);
        uint32_t l1_write_addr = cb.get_write_ptr();
        tt::data_movement::common::noc_async_read_sharded(noc, l1_write_addr, s0, row, 0, page_size);
        noc.async_read_barrier();
        cb.push_back(1);
    }
}
