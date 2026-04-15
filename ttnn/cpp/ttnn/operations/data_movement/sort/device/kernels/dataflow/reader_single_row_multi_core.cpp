// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t input_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_tensor_buffer_addr = get_arg_val<uint32_t>(1);
    const uint32_t coordinator_core_physical_coord_x = get_arg_val<uint32_t>(2);
    const uint32_t coordinator_core_physical_coord_y = get_arg_val<uint32_t>(3);
    const uint32_t coordinator_to_cores_semaphore_id = get_semaphore(get_arg_val<uint32_t>(4));
    const uint32_t cores_to_coordinator_semaphore_id = get_semaphore(get_arg_val<uint32_t>(5));

    // Compile time args
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(4);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(5);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(6);
    constexpr uint32_t number_of_available_cores = get_compile_time_arg_val(7);

    constexpr auto input_tensor_args = TensorAccessorArgs<8>();
    constexpr auto index_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    constexpr uint32_t one_tile = 1;

    // Input tensor config
    constexpr uint32_t tile_size_bytes = get_tile_size(input_tensor_cb_index);
    const auto input_tensor_addr_gen = TensorAccessor(input_tensor_args, input_tensor_buffer_addr, tile_size_bytes);

    // Index tensor config
    const uint32_t index_tensor_output_tile_size_bytes = get_tile_size(index_tensor_cb_index);
    const auto index_tensor_addr_gen =
        TensorAccessor(index_tensor_args, index_tensor_buffer_addr, index_tensor_output_tile_size_bytes);

    // Semaphore setup
    volatile tt_l1_ptr uint32_t* semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(coordinator_to_cores_semaphore_id);
    noc_semaphore_set(semaphore_ptr, VALID);  // Reset the semaphore (Valid - we wait for 0)
    const uint64_t coordinator_core_addr = get_noc_addr(
        coordinator_core_physical_coord_x, coordinator_core_physical_coord_y, cores_to_coordinator_semaphore_id);

    experimental::Noc noc;
    experimental::CircularBuffer cb_input(input_tensor_cb_index);
    experimental::CircularBuffer cb_index(index_tensor_cb_index);

    for (uint32_t h = 0; h < Ht; h++) {
        // Get core start value
        const uint32_t core_start =
            get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Indicate to the coordinator that the core is ready (legacy NOC - uses x,y addressing)
        noc_semaphore_inc(coordinator_core_addr, 1);
        noc_async_atomic_barrier();
        noc_semaphore_wait(semaphore_ptr, 0);     // Wait for coordinator to signal to start
        noc_semaphore_set(semaphore_ptr, VALID);  // Reset the semaphore

        // Processing each row
        uint32_t stages = 0;
        for (uint32_t temp = Wt; temp > 1; temp >>= 1) {
            stages++;
        }

        for (uint32_t stage = 1; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);

                // Wait for coordinator
                noc_semaphore_wait(semaphore_ptr, 0);
                noc_semaphore_set(semaphore_ptr, VALID);  // Reset the semaphore

                uint16_t pair_id = 0;
                uint32_t processing_pair_id = core_start;
                for (uint32_t i = 0; i < Wt; i++) {
                    uint32_t j = i ^ sub_dist;

                    if (j > i) {
                        if (pair_id == processing_pair_id) {
                            // Get indexes of tiles to compare
                            const uint32_t left_tile_id = i;
                            const uint32_t right_tile_id = j;

                            // Read input value data
                            cb_input.reserve_back(one_tile);
                            noc.async_read(
                                input_tensor_addr_gen,
                                cb_input,
                                tile_size_bytes,
                                {.page_id = h * Wt + left_tile_id},
                                {.offset_bytes = 0});
                            noc.async_read_barrier();
                            cb_input.push_back(one_tile);

                            cb_input.reserve_back(one_tile);
                            noc.async_read(
                                input_tensor_addr_gen,
                                cb_input,
                                tile_size_bytes,
                                {.page_id = h * Wt + right_tile_id},
                                {.offset_bytes = 0});
                            noc.async_read_barrier();
                            cb_input.push_back(one_tile);

                            // Read index data
                            cb_index.reserve_back(one_tile);
                            noc.async_read(
                                index_tensor_addr_gen,
                                cb_index,
                                index_tensor_output_tile_size_bytes,
                                {.page_id = h * Wt + left_tile_id},
                                {.offset_bytes = 0});
                            noc.async_read_barrier();
                            cb_index.push_back(one_tile);

                            cb_index.reserve_back(one_tile);
                            noc.async_read(
                                index_tensor_addr_gen,
                                cb_index,
                                index_tensor_output_tile_size_bytes,
                                {.page_id = h * Wt + right_tile_id},
                                {.offset_bytes = 0});
                            noc.async_read_barrier();
                            cb_index.push_back(one_tile);

                            processing_pair_id += number_of_available_cores;
                        }  // if pair_id == processing_pair_id
                        pair_id++;
                    }  // if j > i
                }  // i loop
            }  // sub loop
        }  // stage loop
    }  // h loop
}
