// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_pages = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t write_size = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(dst_args, dst_addr, page_size);

    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_id_out, onepage);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
#ifdef CN_RM
        tt::data_movement::common::noc_async_write_sharded(l1_read_addr, s, i, 0, write_size);
#else
        noc_async_write_page(i, s, l1_read_addr);
#endif

        noc_async_write_barrier();
        cb_pop_front(cb_id_out, onepage);
    }
}
