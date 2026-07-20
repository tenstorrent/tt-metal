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
    const uint32_t cores_to_coordinator_ready_semaphore_arg = get_arg_val<uint32_t>(5);
    const uint32_t cores_to_coordinator_done_semaphore_arg = get_arg_val<uint32_t>(6);
    const uint32_t number_of_dest = get_arg_val<uint32_t>(7);
    const uint32_t input_tensor_buffer_addr = get_arg_val<uint32_t>(8);
    const uint32_t output_tensor_buffer_addr = get_arg_val<uint32_t>(9);
    const uint32_t output_index_tensor_buffer_addr = get_arg_val<uint32_t>(10);

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
    constexpr uint32_t rm_base_offset = output_index_tensor_args.next_compile_time_args_offset();
    constexpr bool is_row_major = get_compile_time_arg_val(rm_base_offset) == 1;
    constexpr uint32_t rm_coord_value_row_cb = get_compile_time_arg_val(rm_base_offset + 1);
    constexpr uint32_t rm_coord_index_row_cb = get_compile_time_arg_val(rm_base_offset + 2);
    constexpr uint32_t W_tile_bytes = get_compile_time_arg_val(rm_base_offset + 3);
    constexpr uint32_t W_index_bytes = get_compile_time_arg_val(rm_base_offset + 4);
    constexpr uint32_t tile_width = get_compile_time_arg_val(rm_base_offset + 5);

    constexpr uint32_t one_tile = 1;
    constexpr uint32_t TILE_H = 32;

    const auto input_tensor_addr_ger = TensorAccessor(input_tensor_args, input_tensor_buffer_addr);
    const auto output_tensor_addr_gen = TensorAccessor(output_tensor_args, output_tensor_buffer_addr);
    const auto output_index_tensor_addr_gen = TensorAccessor(output_index_tensor_args, output_index_tensor_buffer_addr);

    Noc noc;
    CircularBuffer input_tensor_cb(input_tensor_cb_index);
    CircularBuffer index_tensor_cb(index_tensor_cb_index);
    CircularBuffer rm_coord_value_row(rm_coord_value_row_cb);
    CircularBuffer rm_coord_index_row(rm_coord_index_row_cb);
    const uint32_t input_tensor_tile_size = get_tile_size(input_tensor_cb_index);
    const uint32_t index_tensor_tile_size = get_tile_size(index_tensor_cb_index);

    // Semaphore setup
    Semaphore<> coordinator_to_cores_sem(coordinator_to_cores_semaphore_arg);
    // Two separate up-channels from the worker cores: the reader's per-row readiness ->
    // ready sem, the writer's per-pair confirmations -> done sem.  They are kept on
    // distinct semaphores so each exact-match wait() below has its own monotonic target;
    // folded onto one shared counter, at a tile-row boundary (Ht >= 2) a fast reader's
    // next-row readiness could land during the confirmation window and push the counter
    // past the done target, so the wait would never match and the op would deadlock.
    Semaphore<> cores_to_coordinator_ready_sem(cores_to_coordinator_ready_semaphore_arg);
    Semaphore<> cores_to_coordinator_done_sem(cores_to_coordinator_done_semaphore_arg);

    const uint32_t number_of_confirmations = Wt / 2;

    // Copy input data to output and generate index tiles
    for (uint32_t h = 0; h < Ht; h++) {
        // Prepare and move data
        const uint32_t row_base = h * TILE_H;

        for (uint32_t w = 0; w < Wt; w++) {
            // Generate indexes
            if constexpr (is_row_major) {
                for (uint32_t row = 0; row < TILE_H; row++) {
                    rm_coord_value_row.reserve_back(one_tile);
                    noc.async_read(
                        input_tensor_addr_ger,
                        rm_coord_value_row,
                        W_tile_bytes,
                        {.page_id = row_base + row, .offset_bytes = static_cast<uint32_t>(w * W_tile_bytes)},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                    noc.async_write(
                        use<CircularBuffer::AddrSelector::WRITE_PTR>(rm_coord_value_row),
                        output_tensor_addr_gen,
                        W_tile_bytes,
                        {.offset_bytes = 0},
                        {.page_id = row_base + row, .offset_bytes = static_cast<uint32_t>(w * W_tile_bytes)});
                    noc.async_write_barrier();
                    rm_coord_value_row.push_back(one_tile);
                    rm_coord_value_row.pop_front(one_tile);
                }

                rm_coord_index_row.reserve_back(one_tile);
                const uint32_t l1_idx = rm_coord_index_row.get_write_ptr();
                const uint32_t idx_base = w * tile_width;
                if (is_32_bit_data) {
                    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_idx);
                    for (uint32_t c = 0; c < tile_width; c++) {
                        p[c] = idx_base + c;
                    }
                } else {
                    volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_idx);
                    for (uint32_t c = 0; c < tile_width; c++) {
                        p[c] = static_cast<uint16_t>(idx_base + c);
                    }
                }
                for (uint32_t row = 0; row < TILE_H; row++) {
                    noc.async_write(
                        use<CircularBuffer::AddrSelector::WRITE_PTR>(rm_coord_index_row),
                        output_index_tensor_addr_gen,
                        W_index_bytes,
                        {.offset_bytes = 0},
                        {.page_id = row_base + row, .offset_bytes = static_cast<uint32_t>(w * W_index_bytes)});
                    noc.async_write_barrier();
                }
                rm_coord_index_row.push_back(one_tile);
                rm_coord_index_row.pop_front(one_tile);

            } else {
                if (is_32_bit_data) {
                    generate_index_tile<uint32_t>(index_tensor_cb_index, w);
                } else {
                    generate_index_tile<uint16_t>(index_tensor_cb_index, w);
                }

                index_tensor_cb.wait_front(one_tile);
                noc.async_write(
                    index_tensor_cb,
                    output_index_tensor_addr_gen,
                    index_tensor_tile_size,
                    {.offset_bytes = 0},
                    {.page_id = h * Wt + w, .offset_bytes = 0});
                noc.async_write_barrier();
                index_tensor_cb.pop_front(one_tile);

                input_tensor_cb.reserve_back(one_tile);
                noc.async_read(
                    input_tensor_addr_ger,
                    input_tensor_cb,
                    input_tensor_tile_size,
                    {.page_id = h * Wt + w, .offset_bytes = 0},
                    {.offset_bytes = 0});
                noc.async_read_barrier();
                input_tensor_cb.push_back(one_tile);

                input_tensor_cb.wait_front(one_tile);
                noc.async_write(
                    input_tensor_cb,
                    output_tensor_addr_gen,
                    input_tensor_tile_size,
                    {.offset_bytes = 0},
                    {.page_id = h * Wt + w, .offset_bytes = 0});
                noc.async_write_barrier();
                input_tensor_cb.pop_front(one_tile);
            }
        }  // Wt loop

        // Wait until all cores are ready to start
        cores_to_coordinator_ready_sem.wait(number_of_dest);
        cores_to_coordinator_ready_sem.set(0);  // Reset the semaphore

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
                cores_to_coordinator_done_sem.wait(number_of_confirmations);
                cores_to_coordinator_done_sem.set(0);  // Reset the semaphore
            }  // sub loop
        }  // stage loop
    }  // Ht loop
}
