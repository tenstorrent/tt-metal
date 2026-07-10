// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t input_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_tensor_buffer_addr = get_arg_val<uint32_t>(1);
    const uint32_t coordinator_core_physical_coord_x = get_arg_val<uint32_t>(2);
    const uint32_t coordinator_core_physical_coord_y = get_arg_val<uint32_t>(3);
    const uint32_t coordinator_to_cores_semaphore_arg = get_arg_val<uint32_t>(4);
    const uint32_t cores_to_coordinator_semaphore_arg = get_arg_val<uint32_t>(5);

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

    constexpr uint32_t rm_base = index_tensor_args.next_compile_time_args_offset();
    constexpr bool is_row_major = get_compile_time_arg_val(rm_base) == 1;
    constexpr uint32_t rm_input_value_cb_index = get_compile_time_arg_val(rm_base + 1);
    constexpr uint32_t rm_input_index_cb_index = get_compile_time_arg_val(rm_base + 2);
    constexpr uint32_t W_tile_bytes = get_compile_time_arg_val(rm_base + 3);
    constexpr uint32_t W_index_bytes = get_compile_time_arg_val(rm_base + 4);

    constexpr uint32_t one_tile = 1;
    constexpr uint32_t TILE_H = 32;

    const auto input_tensor_addr_gen = TensorAccessor(input_tensor_args, input_tensor_buffer_addr);
    const auto index_tensor_addr_gen = TensorAccessor(index_tensor_args, index_tensor_buffer_addr);

    Noc noc;
    CircularBuffer input_tensor_cb(input_tensor_cb_index);
    CircularBuffer index_tensor_cb(index_tensor_cb_index);
    CircularBuffer rm_input_value_cb(rm_input_value_cb_index);
    CircularBuffer rm_input_index_cb(rm_input_index_cb_index);
    constexpr uint32_t input_tensor_tile_size = get_tile_size(input_tensor_cb_index);
    constexpr uint32_t index_tensor_tile_size = get_tile_size(index_tensor_cb_index);

    // Semaphore setup
    Semaphore<> coordinator_to_cores_sem(coordinator_to_cores_semaphore_arg);
    Semaphore<> cores_to_coordinator_sem(cores_to_coordinator_semaphore_arg);
    coordinator_to_cores_sem.set(VALID);  // Reset the semaphore (Valid - we wait for 0)

    for (uint32_t h = 0; h < Ht; h++) {
        // Get core start value
        const uint32_t core_start =
            get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Indicate to the coordinator that the core is ready
        cores_to_coordinator_sem.up(noc, coordinator_core_physical_coord_x, coordinator_core_physical_coord_y, 1);
        noc.async_atomic_barrier();
        coordinator_to_cores_sem.wait(0);     // Wait for coordinator to signal to start
        coordinator_to_cores_sem.set(VALID);  // Reset the semaphore

        // Processing each row
        uint32_t stages = 0;
        for (uint32_t temp = Wt; temp > 1; temp >>= 1) {
            stages++;
        }

        for (uint32_t stage = 1; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);

                // Wait for coordinator
                coordinator_to_cores_sem.wait(0);
                coordinator_to_cores_sem.set(VALID);  // Reset the semaphore

                uint16_t pair_id = 0;
                uint32_t processing_pair_id = core_start;
                for (uint32_t i = 0; i < Wt; i++) {
                    uint32_t j = i ^ sub_dist;

                    if (j > i) {
                        if (pair_id == processing_pair_id) {
                            // Get indexes of tiles to compare
                            const uint32_t left_tile_id = i;
                            const uint32_t right_tile_id = j;
                            const uint32_t row_base = h * TILE_H;

                            if constexpr (is_row_major) {
                                // Read input value data
                                for (uint32_t tile_id : {left_tile_id, right_tile_id}) {
                                    for (uint32_t row = 0; row < TILE_H; row++) {
                                        rm_input_value_cb.reserve_back(one_tile);
                                        noc.async_read(
                                            input_tensor_addr_gen,
                                            rm_input_value_cb,
                                            W_tile_bytes,
                                            {.page_id = row_base + row,
                                             .offset_bytes = static_cast<uint32_t>(tile_id * W_tile_bytes)},
                                            {.offset_bytes = 0});
                                        noc.async_read_barrier();
                                        rm_input_value_cb.push_back(one_tile);
                                    }
                                    for (uint32_t row = 0; row < TILE_H; row++) {
                                        rm_input_index_cb.reserve_back(one_tile);
                                        noc.async_read(
                                            index_tensor_addr_gen,
                                            rm_input_index_cb,
                                            W_index_bytes,
                                            {.page_id = row_base + row,
                                             .offset_bytes = static_cast<uint32_t>(tile_id * W_index_bytes)},
                                            {.offset_bytes = 0});
                                        noc.async_read_barrier();
                                        rm_input_index_cb.push_back(one_tile);
                                    }
                                }
                            } else {
                                input_tensor_cb.reserve_back(one_tile);
                                noc.async_read(
                                    input_tensor_addr_gen,
                                    input_tensor_cb,
                                    input_tensor_tile_size,
                                    {.page_id = h * Wt + left_tile_id, .offset_bytes = 0},
                                    {.offset_bytes = 0});
                                noc.async_read_barrier();
                                input_tensor_cb.push_back(one_tile);

                                input_tensor_cb.reserve_back(one_tile);
                                noc.async_read(
                                    input_tensor_addr_gen,
                                    input_tensor_cb,
                                    input_tensor_tile_size,
                                    {.page_id = h * Wt + right_tile_id, .offset_bytes = 0},
                                    {.offset_bytes = 0});
                                noc.async_read_barrier();
                                input_tensor_cb.push_back(one_tile);

                                index_tensor_cb.reserve_back(one_tile);
                                noc.async_read(
                                    index_tensor_addr_gen,
                                    index_tensor_cb,
                                    index_tensor_tile_size,
                                    {.page_id = h * Wt + left_tile_id, .offset_bytes = 0},
                                    {.offset_bytes = 0});
                                noc.async_read_barrier();
                                index_tensor_cb.push_back(one_tile);

                                index_tensor_cb.reserve_back(one_tile);
                                noc.async_read(
                                    index_tensor_addr_gen,
                                    index_tensor_cb,
                                    index_tensor_tile_size,
                                    {.page_id = h * Wt + right_tile_id, .offset_bytes = 0},
                                    {.offset_bytes = 0});
                                noc.async_read_barrier();
                                index_tensor_cb.push_back(one_tile);
                            }

                            processing_pair_id += number_of_available_cores;
                        }  // if pair_id == processing_pair_id
                        pair_id++;
                    }  // if j > i
                }  // i loop
            }  // sub loop
        }  // stage loop
    }  // h loop
}
