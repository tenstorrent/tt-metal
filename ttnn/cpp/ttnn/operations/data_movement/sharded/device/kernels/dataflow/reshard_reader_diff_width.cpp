// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
    const uint32_t num_blocks = get_arg_val<uint32_t>(arg_index++);
    const uint32_t output_page_offset = get_arg_val<uint32_t>(arg_index++);

    Noc noc;
    DataflowBuffer dfb(shard_dfb);
    uint32_t l1_write_addr = dfb.get_write_ptr() + output_page_offset * page_size;

    uint32_t mask_byte = 0xff;
    uint32_t mask_short = 0xffff;

    for (uint32_t block_id = 0; block_id < num_blocks; block_id++) {
        const uint32_t num_repeats = get_arg_val<uint32_t>(arg_index++);
        const uint32_t pattern_len = get_arg_val<uint32_t>(arg_index++);

        uint32_t base_pattern_arg_index = arg_index;

        for (uint32_t r = 0; r < num_repeats; r++) {
            // For each repetition, reset the arg pointer to the start of the base pattern
            uint32_t current_pattern_arg_index = base_pattern_arg_index;
            for (uint32_t i = 0; i < pattern_len; i++) {
                const uint32_t meta_stride_core = get_arg_val<uint32_t>(current_pattern_arg_index++);
                const uint32_t meta_stride_data = get_arg_val<uint32_t>(current_pattern_arg_index++);

                const uint32_t meta_stride_x = (meta_stride_core >> 16);
                const uint32_t meta_stride_y = (meta_stride_core & mask_short);

                const uint32_t core_start_stride = get_arg_val<uint32_t>(current_pattern_arg_index++);
                const uint32_t stride_data_offset = get_arg_val<uint32_t>(current_pattern_arg_index++);
                const uint32_t stride_size_num_strides_skip = get_arg_val<uint32_t>(current_pattern_arg_index++);

                const uint32_t base_start_x_index = (core_start_stride >> 24);
                const uint32_t base_start_y_index = (core_start_stride >> 16) & mask_byte;
                const uint32_t stride_x = (core_start_stride >> 8) & mask_byte;
                const uint32_t stride_y = (core_start_stride)&mask_byte;

                const uint32_t num_strides = (stride_size_num_strides_skip >> 8) & mask_byte;
                const bool skip = (stride_size_num_strides_skip & mask_byte) == 1;

                const uint32_t stride_data_in_pages = (stride_data_offset >> 16);
                const uint32_t base_offset_in_pages = (stride_data_offset & mask_short);
                const uint32_t num_pages_per_stride = (stride_size_num_strides_skip >> 16);
                const uint32_t stride_size_in_bytes = num_pages_per_stride * unit_size;

                uint32_t current_start_x_index = base_start_x_index + r * meta_stride_x;
                uint32_t current_start_y_index = base_start_y_index + r * meta_stride_y;
                uint32_t current_offset_in_pages = base_offset_in_pages + r * meta_stride_data;

                uint32_t addr_offset_in_bytes = current_offset_in_pages * unit_size;
                uint32_t core_id_x_index = current_start_x_index;
                uint32_t core_id_y_index = current_start_y_index;

                for (uint32_t stride_idx = 0; stride_idx < num_strides; stride_idx++) {
                    if (!skip) {
                        uint32_t core_id_x = get_arg_val<uint32_t>(core_id_x_index);
                        uint32_t core_id_y = get_arg_val<uint32_t>(y_offset + core_id_y_index);
                        CoreLocalMem<uint32_t> dst(l1_write_addr);
                        noc.async_read(
                            UnicastEndpoint{},
                            dst,
                            stride_size_in_bytes,
                            {.noc_x = core_id_x, .noc_y = core_id_y, .addr = input_shard_addr + addr_offset_in_bytes},
                            {.offset_bytes = 0});
                    }
                    l1_write_addr += stride_size_in_bytes;

                    if (stride_x == 0 and stride_y == 0) {
                        addr_offset_in_bytes += (stride_data_in_pages * unit_size + stride_size_in_bytes);
                    } else {
                        addr_offset_in_bytes += (stride_data_in_pages * unit_size);
                    }
                    core_id_x_index += stride_x;
                    core_id_y_index += stride_y;
                }
            }
        }
        // Advance the main arg_index past the data for this block
        arg_index = base_pattern_arg_index + (pattern_len * 5);
    }
    noc.async_read_barrier();
}
