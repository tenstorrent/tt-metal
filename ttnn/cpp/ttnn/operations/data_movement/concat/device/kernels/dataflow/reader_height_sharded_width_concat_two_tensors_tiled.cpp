// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t input0_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t input1_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t input0_transpose_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t input1_transpose_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t concat_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t output_transpose_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(6);

    constexpr uint32_t input0_num_tiles_height = get_compile_time_arg_val(7);
    constexpr uint32_t input0_num_tiles_width = get_compile_time_arg_val(8);
    constexpr uint32_t input1_num_tiles_height = get_compile_time_arg_val(9);
    constexpr uint32_t input1_num_tiles_width = get_compile_time_arg_val(10);

    constexpr uint32_t output_num_tiles_width = input0_num_tiles_width + input1_num_tiles_width;

    constexpr uint32_t tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t groups = get_compile_time_arg_val(12);

    constexpr uint32_t bf16_tile_size = 32 * 32 * 2;
#ifdef BF8
    constexpr uint32_t input0_stride = bf16_tile_size * input0_num_tiles_width / groups;
    constexpr uint32_t input1_stride = bf16_tile_size * input1_num_tiles_width / groups;
#else
    constexpr uint32_t input0_stride = tile_size * input0_num_tiles_width / groups;
    constexpr uint32_t input1_stride = tile_size * input1_num_tiles_width / groups;
#endif
    constexpr uint32_t group_stride = input0_stride + input1_stride;

    Noc noc;
    CircularBuffer input0_cb(input0_cb_id);
    CircularBuffer input1_cb(input1_cb_id);
    CircularBuffer input0_transpose_cb(input0_transpose_cb_id);
    CircularBuffer input1_transpose_cb(input1_transpose_cb_id);
    CircularBuffer concat_cb(concat_cb_id);

    const uint32_t base_l1_read_addr_0 = input0_transpose_cb.get_read_ptr();
    const uint32_t base_l1_read_addr_1 = input1_transpose_cb.get_read_ptr();
    const uint32_t base_l1_write_addr = concat_cb.get_write_ptr();

    input0_cb.push_back(input0_num_tiles_height * input0_num_tiles_width);
    input1_cb.push_back(input1_num_tiles_height * input1_num_tiles_width);

    for (uint32_t i = 0; i < input0_num_tiles_height; i++) {
        concat_cb.reserve_back(output_num_tiles_width);

        input0_transpose_cb.wait_front(input0_num_tiles_width);

        uint32_t l1_read_addr = base_l1_read_addr_0;
        uint32_t l1_write_addr = base_l1_write_addr;

#ifdef USE_SINGLE_PACKET_READ
        noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            input0_stride,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
             .noc_y = (uint32_t)my_y[noc.get_noc_id()],
             .addr = base_l1_read_addr_0});
        for (uint32_t j = 0; j < groups; j++) {
            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                UnicastEndpoint{},
                dst,
                input0_stride,
                {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                 .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                 .addr = l1_read_addr},
                {.offset_bytes = 0});
            l1_read_addr += input0_stride;
            l1_write_addr += group_stride;
        }
#else
        for (uint32_t j = 0; j < groups; j++) {
            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read(
                UnicastEndpoint{},
                dst,
                input0_stride,
                {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                 .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                 .addr = l1_read_addr},
                {.offset_bytes = 0});
            l1_read_addr += input0_stride;
            l1_write_addr += group_stride;
        }
#endif

        noc.async_read_barrier();
        input0_transpose_cb.pop_front(input0_num_tiles_width);

        input1_transpose_cb.wait_front(input1_num_tiles_width);

        l1_read_addr = base_l1_read_addr_1;
        l1_write_addr = base_l1_write_addr + input0_stride;

#ifdef USE_SINGLE_PACKET_READ
        noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            input1_stride,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
             .noc_y = (uint32_t)my_y[noc.get_noc_id()],
             .addr = base_l1_read_addr_1});
        for (uint32_t j = 0; j < groups; j++) {
            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                UnicastEndpoint{},
                dst,
                input1_stride,
                {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                 .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                 .addr = l1_read_addr},
                {.offset_bytes = 0});
            l1_read_addr += input1_stride;
            l1_write_addr += group_stride;
        }
#else
        for (uint32_t j = 0; j < groups; j++) {
            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read(
                UnicastEndpoint{},
                dst,
                input1_stride,
                {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                 .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                 .addr = l1_read_addr},
                {.offset_bytes = 0});
            l1_read_addr += input1_stride;
            l1_write_addr += group_stride;
        }
#endif

        noc.async_read_barrier();
        input1_transpose_cb.pop_front(input1_num_tiles_width);

        concat_cb.push_back(output_num_tiles_width);
    }
}
