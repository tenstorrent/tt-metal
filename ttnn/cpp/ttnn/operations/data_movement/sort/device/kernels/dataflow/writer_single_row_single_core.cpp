// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

#include "sort_dataflow_common.hpp"

/*
To improve performance of both reader and writer kernels the work has been split so that they both prepare input and
save output data.

Reader:
    * Reads input value data from DRAM and writes it to L1 circular buffer.
    * Write processed index data from L1 to DRAM.

Writer:
    * Generates index input data and writes it to L1 circular buffer.
    * Write output values from L1 to DRAM.
*/
void kernel_main() {
    // Runtime args
    const uint32_t value_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_loop_count = get_arg_val<uint32_t>(1);

    // Compile time args
    constexpr uint32_t value_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(4);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(5);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(6);
    constexpr bool is_32_bit_data = get_compile_time_arg_val(7) == 1;
    constexpr auto value_tensor_args = TensorAccessorArgs<8>();

    // Output tensor config
    constexpr uint32_t one_tile = 1;
    const auto interleaved_accessor0 = TensorAccessor(value_tensor_args, value_tensor_buffer_addr);
    Noc noc;
    CircularBuffer value_tensor_cb(value_tensor_cb_index);
    const uint32_t value_tile_bytes = get_tile_size(value_tensor_cb_index);

    // Move data from L1 to DRAMs
    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        // Calculate tile h coordinate
        const uint32_t h = core_loop * total_number_of_cores +
                           get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Generate index tiles
        for (uint32_t w = 0; w < Wt; w++) {
            if (is_32_bit_data) {
                generate_index_tile<uint32_t>(index_tensor_cb_index, w);
            } else {
                generate_index_tile<uint16_t>(index_tensor_cb_index, w);
            }
        }  // Wt loop

        // Write value tensor to DRAM
        for (uint32_t w = 0; w < Wt; w++) {
            value_tensor_cb.wait_front(one_tile);
            noc.async_write(
                value_tensor_cb,
                interleaved_accessor0,
                value_tile_bytes,
                {.offset_bytes = 0},
                {.page_id = h * Wt + w, .offset_bytes = 0});
            noc.async_write_barrier();
            value_tensor_cb.pop_front(one_tile);
        }  // Wt loop
    }  // core_loop_count loop
}
