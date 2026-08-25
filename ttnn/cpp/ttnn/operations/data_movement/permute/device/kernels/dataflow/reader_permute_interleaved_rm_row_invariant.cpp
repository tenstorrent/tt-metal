// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t N = get_arg(args::N);
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t num_rows = get_arg(args::num_rows);

    const uint32_t start_row = get_arg(args::start_row);
    const uint32_t end_row = get_arg(args::end_row);

    const auto s0 = TensorAccessor(tensor::input);
    DataflowBuffer dfb(dfb::cb_src);
    Noc noc;

    for (uint32_t row = start_row; row < end_row; ++row) {
        dfb.reserve_back(1);
        uint32_t l1_write_addr = dfb.get_write_ptr();
        tt::data_movement::common::noc_async_read_sharded(noc, l1_write_addr, s0, row, 0, page_size);
        noc.async_read_barrier();
        dfb.push_back(1);
    }
}
