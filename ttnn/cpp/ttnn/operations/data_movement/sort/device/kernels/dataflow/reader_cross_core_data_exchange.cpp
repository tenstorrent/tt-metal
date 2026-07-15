// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_core_data_exchange_common.hpp"

#include <cstdint>
#include <utility>

void kernel_main() {
    Noc noc;

    // Runtime args
    const uint32_t input_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_tensor_buffer_addr = get_arg_val<uint32_t>(1);
    const uint32_t physical_core_lookup_table_buffer_addr = get_arg_val<uint32_t>(2);

    // Compile time args
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(0);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(1);
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t index_tensor_output_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t value_tensor_intermediate_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t index_tensor_intermediate_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t value_tensor_peer_cb_index = get_compile_time_arg_val(6);
    constexpr uint32_t index_tensor_peer_cb_index = get_compile_time_arg_val(7);
    constexpr uint32_t physical_core_lookup_table_cb_index = get_compile_time_arg_val(8);

    constexpr uint32_t Ht = get_compile_time_arg_val(9);
    constexpr uint32_t Wt = get_compile_time_arg_val(10);
    constexpr uint32_t number_of_tiles_per_core = get_compile_time_arg_val(11);
    constexpr uint32_t number_of_cores_used = get_compile_time_arg_val(12);
    constexpr bool ascending = get_compile_time_arg_val(13) == 1;

    constexpr uint32_t sem_exchange_id = get_compile_time_arg_val(14);
    constexpr uint32_t sem_barrier_id = get_compile_time_arg_val(15);
    constexpr bool is_row_major = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t rm_input_cb_index = get_compile_time_arg_val(17);
    constexpr uint32_t rm_index_output_cb_index = get_compile_time_arg_val(18);
    constexpr uint32_t W_value_slice_bytes = get_compile_time_arg_val(19);
    constexpr uint32_t W_index_slice_bytes = get_compile_time_arg_val(20);

    constexpr auto input_tensor_args = TensorAccessorArgs<21>();
    constexpr auto index_tensor_output_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();
    constexpr auto physical_core_lookup_table_args =
        TensorAccessorArgs<index_tensor_output_args.next_compile_time_args_offset()>();

    // Constants
    constexpr uint32_t one_tile = 1;
    const uint16_t core_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();
    const uint16_t global_tile_start = core_id * number_of_tiles_per_core;
    const uint16_t global_tile_end = global_tile_start + number_of_tiles_per_core;
    constexpr uint32_t start_core_id = 0;
    constexpr uint32_t leader_core_id = start_core_id;

    // Input tensor config
    constexpr uint32_t input_tensor_tile_size_bytes = get_tile_size(input_tensor_cb_index);
    const auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_buffer_addr);
    CircularBuffer input_tensor_cb(input_tensor_cb_index);
    CircularBuffer rm_input_cb(rm_input_cb_index);

    // Index tensor config
    const uint32_t index_tensor_output_tile_size_bytes = get_tile_size(index_tensor_output_cb_index);
    const auto index_tensor_output_accessor = TensorAccessor(index_tensor_output_args, index_tensor_buffer_addr);
    CircularBuffer index_output_cb(index_tensor_output_cb_index);
    CircularBuffer rm_index_output_cb(rm_index_output_cb_index);

    // Physical core lookup table config
    constexpr uint32_t physical_core_lookup_table_tile_size_bytes = get_tile_size(physical_core_lookup_table_cb_index);
    const auto physical_core_lookup_table_accessor =
        TensorAccessor(physical_core_lookup_table_args, physical_core_lookup_table_buffer_addr);
    CircularBuffer physical_core_lookup_table_cb(physical_core_lookup_table_cb_index);

    // Read lookup table for physical core IDs
    physical_core_lookup_table_cb.reserve_back(one_tile);
    noc.async_read(
        physical_core_lookup_table_accessor,
        physical_core_lookup_table_cb,
        physical_core_lookup_table_tile_size_bytes,
        {.page_id = 0, .offset_bytes = 0},
        {.offset_bytes = 0});
    noc.async_read_barrier();

    // Semaphore setup
    Semaphore<> sem_exchange(sem_exchange_id);
    Semaphore<> sem_barrier(sem_barrier_id);

    // ROW_MAJOR per-core slice byte offset within each input/index DRAM row.
    // Each core owns a contiguous strip of `number_of_tiles_per_core` tiles
    // (= W_value_slice_bytes for values, W_index_slice_bytes for indices).
    constexpr uint32_t TILE_H = 32;  // TILE_HEIGHT
    const uint32_t value_slice_offset_bytes = core_id * W_value_slice_bytes;
    const uint32_t index_slice_offset_bytes = core_id * W_index_slice_bytes;

    for (uint32_t h = 0; h < Ht; h++) {
        // Read input value data
        if constexpr (is_row_major) {
            // ROW_MAJOR input: read TILE_H rows of the per-core W-slice from DRAM
            // into rm_input_cb.  Compute kernel will tilize them into
            // input_tensor_cb_index for the existing TILE-format sort flow.
            const uint32_t row_base = h * TILE_H;
            for (uint32_t row = 0; row < TILE_H; row++) {
                rm_input_cb.reserve_back(one_tile);
                noc.async_read(
                    input_tensor_accessor,
                    rm_input_cb,
                    W_value_slice_bytes,
                    {.page_id = row_base + row, .offset_bytes = value_slice_offset_bytes},
                    {.offset_bytes = 0});
                noc.async_read_barrier();
                rm_input_cb.push_back(one_tile);
            }
        } else {
            // TILE input path
            for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
                input_tensor_cb.reserve_back(one_tile);
                const uint32_t tile_offset = h * Wt + core_id * number_of_tiles_per_core + w;
                noc.async_read(
                    input_tensor_accessor,
                    input_tensor_cb,
                    input_tensor_tile_size_bytes,
                    {.page_id = tile_offset, .offset_bytes = 0},
                    {.offset_bytes = 0});
                noc.async_read_barrier();
                input_tensor_cb.push_back(one_tile);
            }  // w loop
        }

        const uint32_t stages = ilog2(Wt);
        for (uint32_t stage = 2; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                const uint32_t sub_dist = 1 << (sub - 1);

                const uint32_t i = global_tile_start;
                const uint32_t j = i ^ sub_dist;

                if (!(i >= global_tile_start && i < global_tile_end && j >= global_tile_start && j < global_tile_end)) {
                    // Without this barrier, a faster core (in this scenario core C) could start a new exchange
                    // before its peer has finished the previous one, causing a conflict
                    // on the shared semaphore. For example, with three cores A, B, and C:
                    //  A     B     C
                    //  |     |     |
                    //  E <-> E     |   (A and B exchanging tiles)
                    //  E <-> E     |
                    //  E <-> E     |
                    //  E <---E-----|   (C starts exchange with A)
                    //  X     E     |   (A is now in an invalid state)
                    //  X     E     |
                    //
                    // This barrier ensures all cores reach the same stage before proceeding,
                    // preventing such conflicts.
                    sort_barrier(
                        noc,
                        sem_barrier,
                        physical_core_lookup_table_cb_index,
                        core_id,
                        leader_core_id,
                        number_of_cores_used,
                        start_core_id);

                    const uint32_t other_core_id = j / number_of_tiles_per_core;
                    const std::pair<uint32_t, uint32_t> remote_core_physical =
                        get_core_physical_coordinates(other_core_id, physical_core_lookup_table_cb_index);

                    sort_noc_exchange_Wt_tiles(
                        noc,
                        sem_exchange,
                        value_tensor_intermediate_cb_index,
                        index_tensor_intermediate_cb_index,
                        value_tensor_peer_cb_index,
                        index_tensor_peer_cb_index,
                        number_of_tiles_per_core,
                        input_tensor_tile_size_bytes,
                        index_tensor_output_tile_size_bytes,
                        remote_core_physical.first,
                        remote_core_physical.second);
                }  // if !(i >= global_tile_start && i < ...
            }  // sub
        }  // stages

        // Write output index data
        if constexpr (is_row_major) {
            // ROW_MAJOR output indices: drain TILE_H untilized index rows
            // from rm_index_output_cb (compute pack_untilize'd them) and write
            // each row's per-core W-slice back to DRAM.  pack_untilize_block
            // produces little-endian uint16/uint32 elements, so no byte swap.
            const uint32_t row_base = h * TILE_H;
            for (uint32_t row = 0; row < TILE_H; row++) {
                rm_index_output_cb.wait_front(one_tile);
                noc.async_write(
                    rm_index_output_cb,
                    index_tensor_output_accessor,
                    W_index_slice_bytes,
                    {.offset_bytes = 0},
                    {.page_id = row_base + row, .offset_bytes = index_slice_offset_bytes});
                noc.async_write_barrier();
                rm_index_output_cb.pop_front(one_tile);
            }
        } else {
            // Write output index data (TILE path)
            for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
                index_output_cb.wait_front(one_tile);
                const uint32_t tile_offset = h * Wt + core_id * number_of_tiles_per_core + w;
                noc.async_write(
                    index_output_cb,
                    index_tensor_output_accessor,
                    index_tensor_output_tile_size_bytes,
                    {.offset_bytes = 0},
                    {.page_id = tile_offset, .offset_bytes = 0});
                noc.async_write_barrier();
                index_output_cb.pop_front(one_tile);
            }  // Wt loop
        }

    }  // h loop
    physical_core_lookup_table_cb.push_back(one_tile);
}
