// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/noc_traits.h"
///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t scratch_buffer_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t page_size = get_compile_time_arg_val(1);
constexpr uint32_t output_args_cta_idx = 2;
constexpr uint32_t output_args_crta_idx = 0;

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    // Setup Fabric Headers and Connections
    size_t rt_args_idx = 0;
    uint32_t output_base_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t start_page_index = get_arg_val<uint32_t>(rt_args_idx++);  // page start offset
    uint32_t num_pages = get_arg_val<uint32_t>(rt_args_idx++);         // pages for this core

    auto output_addr_gen_args = TensorAccessorArgs<output_args_cta_idx, output_args_crta_idx>();
    auto output_addr_gen = TensorAccessor(output_addr_gen_args, output_base_addr);

    Noc noc_obj;
    DataflowBuffer cb_scratch_buffer(scratch_buffer_cb_id);

    for (uint32_t page_index = start_page_index; page_index < start_page_index + num_pages; ++page_index) {
        cb_scratch_buffer.wait_front(1);
        noc_obj.async_write(cb_scratch_buffer, output_addr_gen, page_size, {}, {.page_id = page_index});
        noc_obj.async_writes_flushed();
        cb_scratch_buffer.pop_front(1);
    }
    noc_obj.async_write_barrier();
}
