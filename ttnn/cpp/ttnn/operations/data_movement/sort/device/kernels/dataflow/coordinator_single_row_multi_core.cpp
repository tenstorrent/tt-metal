// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"

#include "sort_dataflow_common.hpp"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t start_core_physical_coord_x = get_arg_val<uint32_t>(0);
    const uint32_t start_core_physical_coord_y = get_arg_val<uint32_t>(1);
    const uint32_t end_core_physical_coord_x = get_arg_val<uint32_t>(2);
    const uint32_t end_core_physical_coord_y = get_arg_val<uint32_t>(3);
    const uint32_t coordinator_to_cores_semaphore_arg = get_arg_val<uint32_t>(4);
    const uint32_t cores_to_coordinator_semaphore_arg = get_arg_val<uint32_t>(5);
    const uint32_t number_of_dest = get_arg_val<uint32_t>(6);
    const uint32_t input_tensor_buffer_addr = get_arg_val<uint32_t>(7);
    const uint32_t output_tensor_buffer_addr = get_arg_val<uint32_t>(8);
    const uint32_t output_index_tensor_buffer_addr = get_arg_val<uint32_t>(9);

    // Compile time args
    constexpr uint32_t total_work_units = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(3);
    constexpr uint32_t number_of_available_cores = get_compile_time_arg_val(4);
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(6);
    constexpr bool is_32_bit_data = get_compile_time_arg_val(7) == 1;
    constexpr auto input_tensor_args = TensorAccessorArgs<8>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();
    constexpr auto output_index_tensor_args = TensorAccessorArgs<output_tensor_args.next_compile_time_args_offset()>();

    constexpr uint32_t one_tile = 1;

    // Input tensor config
    const auto input_tensor_addr_ger = TensorAccessor(input_tensor_args, input_tensor_buffer_addr);

    // Output tensor config
    const auto output_tensor_addr_gen = TensorAccessor(output_tensor_args, output_tensor_buffer_addr);

    // Output index tensor config
    const auto output_index_tensor_addr_gen = TensorAccessor(output_index_tensor_args, output_index_tensor_buffer_addr);

    Noc noc;
    CircularBuffer input_tensor_cb(input_tensor_cb_index);
    CircularBuffer index_tensor_cb(index_tensor_cb_index);
    const uint32_t input_tile_bytes = get_tile_size(input_tensor_cb_index);
    const uint32_t index_tile_bytes = get_tile_size(index_tensor_cb_index);

    // Semaphore setup
    Semaphore<> coordinator_to_cores_sem(coordinator_to_cores_semaphore_arg);
    Semaphore<> cores_to_coordinator_sem(cores_to_coordinator_semaphore_arg);

    const uint32_t number_of_confirmations = Wt / 2;

    // Copy input data to output and generate index tiles
    for (uint32_t h = 0; h < Ht; h++) {
        // Prepare and move data
        for (uint32_t w = 0; w < Wt; w++) {
            // Generate indexes
            if (is_32_bit_data) {
                generate_index_tile<uint32_t>(index_tensor_cb_index, w);
            } else {
                generate_index_tile<uint16_t>(index_tensor_cb_index, w);
            }

            // Save index tile to output index tensor
            index_tensor_cb.wait_front(one_tile);
            noc.async_write(
                index_tensor_cb,
                output_index_tensor_addr_gen,
                index_tile_bytes,
                {.offset_bytes = 0},
                {.page_id = h * Wt + w, .offset_bytes = 0});
            noc.async_write_barrier();
            index_tensor_cb.pop_front(one_tile);

            // Read input value data
            input_tensor_cb.reserve_back(one_tile);
            noc.async_read(
                input_tensor_addr_ger,
                input_tensor_cb,
                input_tile_bytes,
                {.page_id = h * Wt + w, .offset_bytes = 0},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            input_tensor_cb.push_back(one_tile);

            // Write output value data
            input_tensor_cb.wait_front(one_tile);
            noc.async_write(
                input_tensor_cb,
                output_tensor_addr_gen,
                input_tile_bytes,
                {.offset_bytes = 0},
                {.page_id = h * Wt + w, .offset_bytes = 0});
            noc.async_write_barrier();
            input_tensor_cb.pop_front(one_tile);

        }  // Wt loop

        // Wait until all cores are ready to start
        cores_to_coordinator_sem.wait(number_of_dest);
        cores_to_coordinator_sem.set(0);  // Reset the semaphore

        // Set signal to start processing
        coordinator_to_cores_sem.set_multicast<NocOptions::DEFAULT>(
            noc,
            start_core_physical_coord_x,
            start_core_physical_coord_y,
            end_core_physical_coord_x,
            end_core_physical_coord_y,
            number_of_dest);
        noc.async_write_barrier();

        // Calculate sorting stages
        uint32_t stages = 0;
        for (uint32_t i = Wt; i > 1; i >>= 1) {
            stages++;
        }

        for (uint32_t stage = 1; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                // Set signal to start processing next sub-stage
                coordinator_to_cores_sem.set_multicast<NocOptions::DEFAULT>(
                    noc,
                    start_core_physical_coord_x,
                    start_core_physical_coord_y,
                    end_core_physical_coord_x,
                    end_core_physical_coord_y,
                    number_of_dest);
                noc.async_write_barrier();

                // Wait until cores will process and save data
                cores_to_coordinator_sem.wait(number_of_confirmations);
                cores_to_coordinator_sem.set(0);  // Reset the semaphore
            }  // sub loop
        }  // stage loop
    }  // Ht loop
}
