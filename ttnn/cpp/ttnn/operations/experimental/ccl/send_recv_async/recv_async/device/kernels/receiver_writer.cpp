// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "dataflow_api.h"
#include "tt_metal/hw/inc/accessor/tensor_accessor.h"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t scratch_buffer_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t num_pages = get_compile_time_arg_val(1);
constexpr uint32_t page_size = get_compile_time_arg_val(2);
constexpr uint32_t output_args_cta_idx = 3;
constexpr uint32_t output_args_crta_idx = 0;

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    // Setup Fabric Headers and Connections
    size_t rt_args_idx = 0;
    uint32_t output_base_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t start_page_index = get_arg_val<uint32_t>(rt_args_idx++);

    auto output_addr_gen_args = make_tensor_accessor_args<output_args_cta_idx, output_args_crta_idx>();
    auto output_addr_gen = make_tensor_accessor_from_args(output_addr_gen_args, output_base_addr, page_size);

    for (uint32_t page_index = start_page_index; page_index < start_page_index + num_pages; ++page_index) {
        cb_wait_front(scratch_buffer_cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(scratch_buffer_cb_id);
        auto noc_write_addr = output_addr_gen.get_noc_addr(page_index);
        noc_async_write<page_size>(l1_read_addr, noc_write_addr, page_size);
        noc_async_writes_flushed();
        cb_pop_front(scratch_buffer_cb_id, 1);
    }
    noc_async_write_barrier();
}
