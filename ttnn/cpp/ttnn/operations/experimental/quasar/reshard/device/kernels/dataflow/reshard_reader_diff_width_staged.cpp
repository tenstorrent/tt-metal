// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Staged generic reshard reader (differing input/output page size; compressed stride blocks).
//
// Gathers this core's output shard from the input shards into the staging DFB, one unit_size unit
// (the gcd of the input and output page sizes) per entry, so each TXN_ID read is one full entry. Every
// thread walks the whole stride pattern; thread t of N reads the output units t, t + N, ... in order,
// which is the order the strided DFB hands them to the writers. The host uses this kernel only when no
// stride is a skip stride, so output unit k lands at offset k * unit_size of the shard.
//
// Varargs: [0 .. num_x_cores + num_y_cores) physical core-coordinate table (random-indexed), then
// num_output_pages, num_blocks, output_page_offset (unused, always 0), followed by compressed stride
// blocks.

#include <stdint.h>
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_x_cores = get_arg(args::num_x_cores);
    constexpr uint32_t num_y_cores = get_arg(args::num_y_cores);
    constexpr uint32_t unit_size = get_arg(args::unit_size);

    uint32_t y_offset = num_x_cores;

    uint32_t arg_index = num_x_cores + num_y_cores;
    TensorAccessor input(tensor::input);
    const uint32_t input_shard_addr = input.get_bank_base_address();
    arg_index++;  // num_output_pages
    const uint32_t num_blocks = get_vararg(arg_index++);
    arg_index++;  // output_page_offset

    Noc noc;
    DataflowBuffer cb(dfb::stage);
    const uint32_t thread_id = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();
    uint32_t unit = 0;

    uint32_t mask_byte = 0xff;
    uint32_t mask_short = 0xffff;

    for (uint32_t block_id = 0; block_id < num_blocks; block_id++) {
        const uint32_t num_repeats = get_vararg(arg_index++);
        const uint32_t pattern_len = get_vararg(arg_index++);

        uint32_t base_pattern_arg_index = arg_index;

        for (uint32_t r = 0; r < num_repeats; r++) {
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
                    const uint32_t core_id_x = get_vararg(core_id_x_index);
                    const uint32_t core_id_y = get_vararg(y_offset + core_id_y_index);
                    for (uint32_t u = 0; u < num_pages_per_stride; u++, unit++) {
                        if (unit % num_threads == thread_id) {
                            noc.async_read<NocOptions::TXN_ID>(
                                UnicastEndpoint{},
                                cb,
                                {.noc_x = core_id_x,
                                 .noc_y = core_id_y,
                                 .addr = input_shard_addr + addr_offset_in_bytes + u * unit_size},
                                {});
                        }
                    }

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
        arg_index = base_pattern_arg_index + (pattern_len * 5);
    }
}
