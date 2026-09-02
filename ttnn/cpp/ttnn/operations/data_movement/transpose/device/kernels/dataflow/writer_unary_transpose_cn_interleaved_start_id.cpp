// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_pages = get_arg(args::num_pages);
    uint32_t start_id = get_arg(args::start_id);

    constexpr auto page_size = get_arg(args::page_size);
    constexpr auto write_size = get_arg(args::write_size);

    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb(dfb::in0);

    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb.wait_front(onepage);
#ifdef CN_RM
        // Restored native sharded multi-page split (see common.hpp helper).
        // RM-only: see reader kernel for rationale; TILE-layout below uses original API.
        const uint32_t dfb_read_ptr = dfb.get_read_ptr();
        tt::data_movement::common::noc_async_write_sharded(noc, dfb_read_ptr, s, i, 0, write_size);
#else
        noc.async_write(dfb, s, page_size, {.offset_bytes = 0}, {.page_id = i});
#endif
        noc.async_write_barrier();
        dfb.pop_front(onepage);
    }
}
