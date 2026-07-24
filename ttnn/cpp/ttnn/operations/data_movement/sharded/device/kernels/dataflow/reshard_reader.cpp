// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t shard_dfb = get_compile_time_arg_val(0);
    constexpr uint32_t num_x_cores = get_compile_time_arg_val(1);
    constexpr uint32_t num_y_cores = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t unit_size = get_compile_time_arg_val(4);

    uint32_t y_offset = num_x_cores;

    uint32_t arg_index = num_x_cores + num_y_cores;
    const uint32_t input_shard_addr = get_arg_val<uint32_t>(arg_index++);
    const uint32_t num_output_pages = get_arg_val<uint32_t>(arg_index++);
    const uint32_t num_ranges = get_arg_val<uint32_t>(arg_index++);
    const uint32_t output_page_offset = get_arg_val<uint32_t>(arg_index++);

    Noc noc;
    DataflowBuffer dfb(shard_dfb);
    uint32_t l1_write_addr = dfb.get_write_ptr() + output_page_offset * page_size;

    uint32_t mask_byte = 0x0ff;     // 8 bits
    uint32_t mask_short = 0x0ffff;  // 16 bits

    for (uint32_t range_id = 0; range_id < num_ranges; range_id++) {
        const uint32_t core_start_stride = get_arg_val<uint32_t>(arg_index++);
        const uint32_t start_x_index = (core_start_stride >> 24);
        const uint32_t start_y_index = (core_start_stride >> 16) & mask_byte;
        const uint32_t stride_x = (core_start_stride >> 8) & mask_byte;
        const uint32_t stride_y = (core_start_stride)&mask_byte;
        const uint32_t start_x = get_arg_val<uint32_t>(start_x_index);
        const uint32_t start_y = get_arg_val<uint32_t>(y_offset + start_y_index);

        const uint32_t stride_data_offset = get_arg_val<uint32_t>(arg_index++);
        const uint32_t stride_size_num_strides_skip = get_arg_val<uint32_t>(arg_index++);
        const uint32_t num_strides = ((stride_size_num_strides_skip)&mask_short) >> 8;
        const bool skip = (((stride_size_num_strides_skip)&mask_byte) == 1);

        const uint32_t stride_data = ((stride_data_offset >> 16)) * unit_size;
        const uint32_t offset = ((stride_data_offset)&mask_short) * unit_size;
        const uint32_t num_pages_per_stride = (stride_size_num_strides_skip >> 16);
        const uint32_t stride_size = num_pages_per_stride * unit_size;

        uint32_t addr_offset = offset;
        uint32_t core_id_x_index = start_x_index;
        uint32_t core_id_y_index = start_y_index;

        for (uint32_t stride_idx = 0; stride_idx < num_strides; stride_idx++) {
            if (!skip) {
                uint32_t core_id_x = get_arg_val<uint32_t>(core_id_x_index);
                uint32_t core_id_y = get_arg_val<uint32_t>(y_offset + core_id_y_index);
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read(
                    UnicastEndpoint{},
                    dst,
                    stride_size,
                    {.noc_x = core_id_x, .noc_y = core_id_y, .addr = input_shard_addr + addr_offset},
                    {.offset_bytes = 0});
                l1_write_addr += stride_size;
            } else {
                l1_write_addr += stride_size;
            }
            if (stride_x == 0 and stride_y == 0) {
                addr_offset += (stride_data + stride_size);
            } else {
                addr_offset += (stride_data);
            }
            core_id_x_index += stride_x;
            core_id_y_index += stride_y;
        }
    }
    noc.async_read_barrier();
}
