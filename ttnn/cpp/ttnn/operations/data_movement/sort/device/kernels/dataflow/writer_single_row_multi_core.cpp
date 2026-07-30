// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

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

    // The sort runs in place in the two output tensors, so the writer's destinations are the same
    // buffers the reader draws its operands from.
    const auto input_tensor_addr_gen = TensorAccessor(tensor::input_tensor);
    const auto index_tensor_addr_gen = TensorAccessor(tensor::index_tensor);

    Noc noc;
#ifdef IS_ROW_MAJOR
    DataflowBuffer rm_output_value_dfb(dfb::rm_output_value);
    DataflowBuffer rm_output_index_dfb(dfb::rm_output_index);
#else
    DataflowBuffer input_output_dfb(dfb::input_tensor_output);
    DataflowBuffer index_output_dfb(dfb::index_tensor_output);
    const uint32_t input_tensor_tile_size = input_output_dfb.get_tile_size();
    const uint32_t index_tensor_tile_size = index_output_dfb.get_tile_size();
#endif

    // Semaphore setup
    Semaphore<> cores_to_coordinator_done_sem(sem::cores_to_coordinator_done);

    for (uint32_t h = 0; h < Ht; h++) {
        // Get core start value
        const uint32_t core_start =
            get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Processing each row
        uint32_t stages = 0;
        for (uint32_t temp = Wt; temp > 1; temp >>= 1) {
            stages++;
        }

        for (uint32_t stage = 1; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);

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

                            for (uint32_t tile_id : {left_tile_id, right_tile_id}) {
                                for (uint32_t row = 0; row < TILE_H; row++) {
                                    rm_output_index_dfb.wait_front(one_tile);
                                    noc.async_write(
                                        rm_output_index_dfb,
                                        index_tensor_addr_gen,
                                        W_index_bytes,
                                        {.offset_bytes = 0},
                                        {.page_id = row_base + row,
                                         .offset_bytes = static_cast<uint32_t>(tile_id * W_index_bytes)});
                                    noc.async_write_barrier();
                                    rm_output_index_dfb.pop_front(one_tile);
                                }
                                for (uint32_t row = 0; row < TILE_H; row++) {
                                    rm_output_value_dfb.wait_front(one_tile);
                                    noc.async_write(
                                        rm_output_value_dfb,
                                        input_tensor_addr_gen,
                                        W_tile_bytes,
                                        {.offset_bytes = 0},
                                        {.page_id = row_base + row,
                                         .offset_bytes = static_cast<uint32_t>(tile_id * W_tile_bytes)});
                                    noc.async_write_barrier();
                                    rm_output_value_dfb.pop_front(one_tile);
                                }
                            }
#else
                            index_output_dfb.wait_front(one_tile);
                            noc.async_write(
                                index_output_dfb,
                                index_tensor_addr_gen,
                                index_tensor_tile_size,
                                {.offset_bytes = 0},
                                {.page_id = h * Wt + left_tile_id, .offset_bytes = 0});
                            noc.async_write_barrier();
                            index_output_dfb.pop_front(one_tile);

                            index_output_dfb.wait_front(one_tile);
                            noc.async_write(
                                index_output_dfb,
                                index_tensor_addr_gen,
                                index_tensor_tile_size,
                                {.offset_bytes = 0},
                                {.page_id = h * Wt + right_tile_id, .offset_bytes = 0});
                            noc.async_write_barrier();
                            index_output_dfb.pop_front(one_tile);

                            input_output_dfb.wait_front(one_tile);
                            noc.async_write(
                                input_output_dfb,
                                input_tensor_addr_gen,
                                input_tensor_tile_size,
                                {.offset_bytes = 0},
                                {.page_id = h * Wt + left_tile_id, .offset_bytes = 0});
                            noc.async_write_barrier();
                            input_output_dfb.pop_front(one_tile);

                            input_output_dfb.wait_front(one_tile);
                            noc.async_write(
                                input_output_dfb,
                                input_tensor_addr_gen,
                                input_tensor_tile_size,
                                {.offset_bytes = 0},
                                {.page_id = h * Wt + right_tile_id, .offset_bytes = 0});
                            noc.async_write_barrier();
                            input_output_dfb.pop_front(one_tile);
#endif

                            // Signalize readiness to the coordinator
                            cores_to_coordinator_done_sem.up(
                                noc, coordinator_core_physical_coord_x, coordinator_core_physical_coord_y, 1);
                            noc.async_atomic_barrier();

                            processing_pair_id += number_of_available_cores;
                        }  // if pair_id == processing_pair_id
                        pair_id++;
                    }  // if j > i
                }  // i loop
            }  // sub loop
        }  // stage loop
    }  // h loop
    noc.async_atomic_barrier();
}
