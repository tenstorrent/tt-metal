// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <api/debug/dprint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t input0_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t input1_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t input0_transpose_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t input1_transpose_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t concat_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t output_transpose_dfb_id = get_compile_time_arg_val(5);
    constexpr uint32_t output_dfb_id = get_compile_time_arg_val(6);

    constexpr uint32_t input0_num_tiles_height = get_compile_time_arg_val(7);
    constexpr uint32_t input0_num_tiles_width = get_compile_time_arg_val(8);
    constexpr uint32_t input1_num_tiles_height = get_compile_time_arg_val(9);
    constexpr uint32_t input1_num_tiles_width = get_compile_time_arg_val(10);

    constexpr uint32_t tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t groups = get_compile_time_arg_val(12);

    constexpr uint32_t input0_stride = tile_size * input0_num_tiles_width / groups;
    constexpr uint32_t input1_stride = tile_size * input1_num_tiles_width / groups;

    constexpr uint32_t width_len_bytes = tile_size * (input0_num_tiles_width + input1_num_tiles_width);

    Noc noc;
    DataflowBuffer output_dfb(output_dfb_id);
    DataflowBuffer output_transpose_dfb(output_transpose_dfb_id);

    const uint32_t base_l1_write_addr = output_dfb.get_write_ptr();
    uint32_t l1_write_addr = base_l1_write_addr;
    for (uint32_t i = 0; i < input0_num_tiles_height; i++) {
        output_dfb.reserve_back(input0_num_tiles_width + input1_num_tiles_width);
        output_transpose_dfb.wait_front(input0_num_tiles_width + input1_num_tiles_width);

        const uint32_t base_l1_read_addr_0 = output_transpose_dfb.get_read_ptr();
        CoreLocalMem<uint32_t> dst(l1_write_addr);
        noc.async_read(
            UnicastEndpoint{},
            dst,
            width_len_bytes,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
             .noc_y = (uint32_t)my_y[noc.get_noc_id()],
             .addr = base_l1_read_addr_0},
            {.offset_bytes = 0});
        l1_write_addr += width_len_bytes;

        noc.async_read_barrier();

        output_transpose_dfb.pop_front(input0_num_tiles_width + input1_num_tiles_width);
        output_dfb.push_back(input0_num_tiles_width + input1_num_tiles_width);
    }
}
