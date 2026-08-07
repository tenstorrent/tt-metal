// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t coordinator_core_physical_coord_x = get_arg(args::coordinator_core_physical_coord_x);
    const uint32_t coordinator_core_physical_coord_y = get_arg(args::coordinator_core_physical_coord_y);

    // Compile time args
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t total_number_of_cores = get_arg(args::total_number_of_cores);
    constexpr uint32_t compute_with_storage_grid_size_x = get_arg(args::compute_with_storage_grid_size_x);
    constexpr uint32_t compute_with_storage_grid_size_y = get_arg(args::compute_with_storage_grid_size_y);
    constexpr uint32_t number_of_available_cores = get_arg(args::number_of_available_cores);
    constexpr uint32_t W_tile_bytes = get_arg(args::W_tile_bytes);
    constexpr uint32_t W_index_bytes = get_arg(args::W_index_bytes);

    constexpr uint32_t one_tile = 1;
    constexpr uint32_t TILE_H = 32;

    // The coordinator has already staged the input into the value-output buffer and generated the
    // index tensor beside it, so a worker reads both of its operands from the output tensors.
    const auto input_tensor_addr_gen = TensorAccessor(tensor::input_tensor);
    const auto index_tensor_addr_gen = TensorAccessor(tensor::index_tensor);

    Noc noc;
#ifdef IS_ROW_MAJOR
    DataflowBuffer rm_input_value_dfb(dfb::rm_input_value);
    DataflowBuffer rm_input_index_dfb(dfb::rm_input_index);
#else
    DataflowBuffer input_tensor_dfb(dfb::input_tensor);
    DataflowBuffer index_tensor_dfb(dfb::index_tensor);
    const uint32_t input_tensor_tile_size = input_tensor_dfb.get_tile_size();
    const uint32_t index_tensor_tile_size = index_tensor_dfb.get_tile_size();
#endif

    // Semaphore setup
    Semaphore<> coordinator_to_cores_sem(sem::coordinator_to_cores);
    Semaphore<> cores_to_coordinator_ready_sem(sem::cores_to_coordinator_ready);
    coordinator_to_cores_sem.set(VALID);  // Reset the semaphore (Valid - we wait for 0)

    for (uint32_t h = 0; h < Ht; h++) {
        // Get core start value
        const uint32_t core_start =
            get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Indicate to the coordinator that the core is ready
        cores_to_coordinator_ready_sem.up(noc, coordinator_core_physical_coord_x, coordinator_core_physical_coord_y, 1);
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
#ifdef IS_ROW_MAJOR
                            const uint32_t row_base = h * TILE_H;

                            // Read input value data
                            for (uint32_t tile_id : {left_tile_id, right_tile_id}) {
                                for (uint32_t row = 0; row < TILE_H; row++) {
                                    rm_input_value_dfb.reserve_back(one_tile);
                                    noc.async_read(
                                        input_tensor_addr_gen,
                                        rm_input_value_dfb,
                                        W_tile_bytes,
                                        {.page_id = row_base + row,
                                         .offset_bytes = static_cast<uint32_t>(tile_id * W_tile_bytes)},
                                        {.offset_bytes = 0});
                                    noc.async_read_barrier();
                                    rm_input_value_dfb.push_back(one_tile);
                                }
                                for (uint32_t row = 0; row < TILE_H; row++) {
                                    rm_input_index_dfb.reserve_back(one_tile);
                                    noc.async_read(
                                        index_tensor_addr_gen,
                                        rm_input_index_dfb,
                                        W_index_bytes,
                                        {.page_id = row_base + row,
                                         .offset_bytes = static_cast<uint32_t>(tile_id * W_index_bytes)},
                                        {.offset_bytes = 0});
                                    noc.async_read_barrier();
                                    rm_input_index_dfb.push_back(one_tile);
                                }
                            }
#else
                            input_tensor_dfb.reserve_back(one_tile);
                            noc.async_read(
                                input_tensor_addr_gen,
                                input_tensor_dfb,
                                input_tensor_tile_size,
                                {.page_id = h * Wt + left_tile_id, .offset_bytes = 0},
                                {.offset_bytes = 0});
                            noc.async_read_barrier();
                            input_tensor_dfb.push_back(one_tile);

                            input_tensor_dfb.reserve_back(one_tile);
                            noc.async_read(
                                input_tensor_addr_gen,
                                input_tensor_dfb,
                                input_tensor_tile_size,
                                {.page_id = h * Wt + right_tile_id, .offset_bytes = 0},
                                {.offset_bytes = 0});
                            noc.async_read_barrier();
                            input_tensor_dfb.push_back(one_tile);

                            index_tensor_dfb.reserve_back(one_tile);
                            noc.async_read(
                                index_tensor_addr_gen,
                                index_tensor_dfb,
                                index_tensor_tile_size,
                                {.page_id = h * Wt + left_tile_id, .offset_bytes = 0},
                                {.offset_bytes = 0});
                            noc.async_read_barrier();
                            index_tensor_dfb.push_back(one_tile);

                            index_tensor_dfb.reserve_back(one_tile);
                            noc.async_read(
                                index_tensor_addr_gen,
                                index_tensor_dfb,
                                index_tensor_tile_size,
                                {.page_id = h * Wt + right_tile_id, .offset_bytes = 0},
                                {.offset_bytes = 0});
                            noc.async_read_barrier();
                            index_tensor_dfb.push_back(one_tile);
#endif

                            processing_pair_id += number_of_available_cores;
                        }  // if pair_id == processing_pair_id
                        pair_id++;
                    }  // if j > i
                }  // i loop
            }  // sub loop
        }  // stage loop
    }  // h loop
}
