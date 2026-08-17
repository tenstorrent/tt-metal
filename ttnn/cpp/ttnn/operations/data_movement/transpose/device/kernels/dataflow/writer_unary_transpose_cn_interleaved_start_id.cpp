// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_pages = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t dfb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t write_size = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    DataflowBuffer dfb(dfb_id_out);

    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb.wait_front(onepage);
#ifdef CN_RM
        // Restored native sharded multi-page split (see common.hpp helper).
        // RM-only: see reader kernel for rationale; TILE-layout below uses original API.
        const uint32_t cb_read_ptr = dfb.get_read_ptr();
        tt::data_movement::common::noc_async_write_sharded(noc, cb_read_ptr, s, i, 0, write_size);
#else
        noc.async_write(dfb, s, page_size, {.offset_bytes = 0}, {.page_id = i});
#endif
        noc.async_write_barrier();
        dfb.pop_front(onepage);
    }
}
