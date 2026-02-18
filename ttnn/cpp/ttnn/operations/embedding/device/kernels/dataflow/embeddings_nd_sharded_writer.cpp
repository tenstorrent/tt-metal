// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "ttnn/kernel/kernel_utils.hpp"

void kernel_main() {
    uint32_t output_buffer_src_addr = get_arg_val<uint32_t>(0);
    uint32_t input_page_id = get_arg_val<uint32_t>(1);
    uint32_t num_of_pages = get_arg_val<uint32_t>(2);

    constexpr uint32_t output_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t weight_page_size = get_compile_time_arg_val(1);
    constexpr uint32_t elems_per_page = get_compile_time_arg_val(2);

    constexpr auto output_args = TensorAccessorArgs<3>();
    const auto output = TensorAccessor(output_args, output_buffer_src_addr, weight_page_size);

    for (uint32_t page_id = input_page_id; page_id < input_page_id + num_of_pages; page_id++) {
        auto flat_input_idx = page_id * elems_per_page;
        auto output_page_id = flat_input_idx;
        auto output_pages = output.pages(output_page_id, output_page_id + elems_per_page);
        auto output_page_iter = output_pages.begin();

        for (uint32_t index = 0; index < elems_per_page; ++index, ++output_page_iter) {
            cb_wait_front(output_cb_index, 1);
            uint32_t output_cb_addr = get_read_ptr(output_cb_index);

            noc_async_write<weight_page_size>(output_cb_addr, output_page_iter->noc_addr(), weight_page_size);
            noc_async_write_barrier();

            cb_pop_front(output_cb_index, 1);
        }
    }
}
