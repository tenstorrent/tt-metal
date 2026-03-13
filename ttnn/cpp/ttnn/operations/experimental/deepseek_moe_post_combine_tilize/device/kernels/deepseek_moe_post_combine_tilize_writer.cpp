// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tilize_output_cb_id = get_named_compile_time_arg_val("tilize_output_cb_id");
    constexpr uint32_t output_tile_page_size = get_named_compile_time_arg_val("output_tile_page_size");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("num_tiles");

    uint32_t rt_args_idx = 0;
    uint32_t output_tile_page_offset = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(arg_idx++);

    constexpr auto output_tensor_accessor_args = TensorAccessorArgs<0>();
    const auto output_tensor_accessor =
        TensorAccessor(output_tensor_accessor_args, output_tensor_address, output_tile_page_size);

    cb_wait_front(tilize_output_cb_id, num_tiles);
    uint32_t l1_read_addr = get_read_ptr(tilize_output_cb_id);

    uint32_t page_id = output_tile_offset;
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        uint64_t noc_addr =
            get_noc_addr(page_id, output_tensor_accessor) noc_async_write(noc_addr, l1_read_addr, tile_page_size)

                l1_read_addr += tile_page_size;
        page_id++;
    }

    noc_async_write_barrier();
    cb_pop_front(tilize_output_cb_id, num_tiles);
}
