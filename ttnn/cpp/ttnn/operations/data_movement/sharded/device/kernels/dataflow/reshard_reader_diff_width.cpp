// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Generic reshard reader for differing input/output page sizes (stride-of-strides / compressed blocks).
//   - The input base L1 address comes from TensorAccessor(tensor::input).get_bank_base_address() (a Case 2
//     raw-pointer binding); the raw noc.async_read walk is otherwise unchanged.
//   - The output buffer's local L1 base comes from DataflowBuffer(dfb::shard_cb).get_write_ptr(); the DFB
//     borrows the output tensor's buffer and is used here purely as an address source.
//   - Per-core block data is positional varargs. Layout (the host drops the legacy input-addr slot):
//       [0 .. num_x_cores + num_y_cores)   physical core-coordinate table (random-indexed)
//       then: num_output_pages, num_blocks, output_page_offset, followed by the compressed stride blocks.

#include <stdint.h>
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_x_cores = get_arg(args::num_x_cores);
    constexpr uint32_t num_y_cores = get_arg(args::num_y_cores);
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t unit_size = get_arg(args::unit_size);

    uint32_t y_offset = num_x_cores;

    TensorAccessor input(tensor::input);
    const uint32_t input_shard_addr = input.get_bank_base_address();

    uint32_t arg_index = num_x_cores + num_y_cores;
    const uint32_t num_output_pages = get_vararg(arg_index++);
    const uint32_t num_blocks = get_vararg(arg_index++);
    const uint32_t output_page_offset = get_vararg(arg_index++);

    Noc noc;
    DataflowBuffer cb(dfb::shard_cb);
    uint32_t l1_write_addr = cb.get_write_ptr() + output_page_offset * page_size;

    uint32_t mask_byte = 0xff;
    uint32_t mask_short = 0xffff;

    for (uint32_t block_id = 0; block_id < num_blocks; block_id++) {
        const uint32_t num_repeats = get_vararg(arg_index++);
        const uint32_t pattern_len = get_vararg(arg_index++);

        uint32_t base_pattern_arg_index = arg_index;

        for (uint32_t r = 0; r < num_repeats; r++) {
            // For each repetition, reset the arg pointer to the start of the base pattern
            uint32_t current_pattern_arg_index = base_pattern_arg_index;
            for (uint32_t i = 0; i < pattern_len; i++) {
                const uint32_t meta_stride_core = get_vararg(current_pattern_arg_index++);
                const uint32_t meta_stride_data = get_vararg(current_pattern_arg_index++);

                const uint32_t meta_stride_x = (meta_stride_core >> 16);
                const uint32_t meta_stride_y = (meta_stride_core & mask_short);

                const uint32_t core_start_stride = get_vararg(current_pattern_arg_index++);
                const uint32_t stride_data_offset = get_vararg(current_pattern_arg_index++);
                const uint32_t stride_size_num_strides_skip = get_vararg(current_pattern_arg_index++);

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
                        uint32_t core_id_x = get_vararg(core_id_x_index);
                        uint32_t core_id_y = get_vararg(y_offset + core_id_y_index);
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
