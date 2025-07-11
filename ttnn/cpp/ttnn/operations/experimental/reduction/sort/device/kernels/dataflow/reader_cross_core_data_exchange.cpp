// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

#include "cross_core_data_exchange_common.hpp"

#include <cstdint>
#include <utility>

void kernel_main() {
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

    constexpr bool input_tensor_is_dram = get_compile_time_arg_val(9) == 1;
    constexpr bool index_tensor_output_is_dram = get_compile_time_arg_val(10) == 1;
    constexpr bool physical_core_lookup_table_is_dram = get_compile_time_arg_val(11) == 1;
    constexpr uint32_t Ht = get_compile_time_arg_val(12);
    constexpr uint32_t Wt = get_compile_time_arg_val(13);
    constexpr uint32_t number_of_tiles_per_core = get_compile_time_arg_val(14);
    constexpr uint32_t number_of_cores_used = get_compile_time_arg_val(15);
    constexpr bool ascending = get_compile_time_arg_val(16) == 1;

    const uint32_t sem_exchange_addr = get_semaphore(get_compile_time_arg_val(17));
    const uint32_t sem_barrier_addr = get_semaphore(get_compile_time_arg_val(18));

    // Constants
    constexpr uint32_t one_tile = 1;
    const uint16_t core_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();
    const uint16_t global_tile_start = core_id * number_of_tiles_per_core;
    const uint16_t global_tile_end = global_tile_start + number_of_tiles_per_core;
    constexpr uint32_t start_core_id = 0;
    constexpr uint32_t leader_core_id = start_core_id;

    // Input tensor config
    constexpr uint32_t input_tensor_tile_size_bytes = get_tile_size(input_tensor_cb_index);
    constexpr DataFormat input_tensor_data_format = get_dataformat(input_tensor_cb_index);
    const InterleavedAddrGenFast<input_tensor_is_dram> input_tensor_accessor = {
        .bank_base_address = input_tensor_buffer_addr,
        .page_size = input_tensor_tile_size_bytes,
        .data_format = input_tensor_data_format};

    // Index tensor config
    const uint32_t index_tensor_output_tile_size_bytes = get_tile_size(index_tensor_output_cb_index);
    const DataFormat index_tensor_output_data_format = get_dataformat(index_tensor_output_cb_index);
    const InterleavedAddrGenFast<index_tensor_output_is_dram> index_tensor_output_accessor = {
        .bank_base_address = index_tensor_buffer_addr,
        .page_size = index_tensor_output_tile_size_bytes,
        .data_format = index_tensor_output_data_format};

    // Physical core lookup table config
    constexpr uint32_t physical_core_lookup_table_tile_size_bytes = get_tile_size(physical_core_lookup_table_cb_index);
    constexpr DataFormat physical_core_lookup_table_data_format = get_dataformat(physical_core_lookup_table_cb_index);
    const InterleavedAddrGenFast<physical_core_lookup_table_is_dram> physical_core_lookup_table_accessor = {
        .bank_base_address = physical_core_lookup_table_buffer_addr,
        .page_size = physical_core_lookup_table_tile_size_bytes,
        .data_format = physical_core_lookup_table_data_format};

    // Read lookup table for physical core IDs
    cb_reserve_back(physical_core_lookup_table_cb_index, one_tile);
    const uint32_t physical_core_lookup_table_l1_write_addr = get_write_ptr(physical_core_lookup_table_cb_index);
    uint64_t noc_addr = get_noc_addr(0, physical_core_lookup_table_accessor);
    noc_async_read(noc_addr, physical_core_lookup_table_l1_write_addr, physical_core_lookup_table_tile_size_bytes);
    noc_async_read_barrier();

    // Semaphore setup
    sem_ptr_t sem_self_exchange_ptr = reinterpret_cast<sem_ptr_t>(sem_exchange_addr);

    for (uint32_t h = 0; h < Ht; h++) {
        // Read input value data
        for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
            cb_reserve_back(input_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr = get_write_ptr(input_tensor_cb_index);
            const uint32_t tile_offset = h * Wt + core_id * number_of_tiles_per_core + w;
            noc_async_read_tile(tile_offset, input_tensor_accessor, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(input_tensor_cb_index, one_tile);
        }  // w loop

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
                        physical_core_lookup_table_cb_index,
                        sem_barrier_addr,
                        core_id,
                        leader_core_id,
                        number_of_cores_used,
                        start_core_id);

                    const uint32_t other_core_id = j / number_of_tiles_per_core;
                    const std::pair<uint32_t, uint32_t> remote_core_physical =
                        get_core_physical_coordinates(other_core_id, physical_core_lookup_table_cb_index);

                    sort_noc_exchange_Wt_tiles(
                        value_tensor_intermediate_cb_index,
                        index_tensor_intermediate_cb_index,
                        value_tensor_peer_cb_index,
                        index_tensor_peer_cb_index,
                        number_of_tiles_per_core,
                        input_tensor_tile_size_bytes,
                        index_tensor_output_tile_size_bytes,
                        remote_core_physical.first,
                        remote_core_physical.second,
                        sem_self_exchange_ptr);
                }  // if !(i >= global_tile_start && i < ...
            }  // sub
        }  // stages

        // Write output index data
        for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
            cb_wait_front(index_tensor_output_cb_index, one_tile);
            const uint32_t l1_write_addr_index = get_read_ptr(index_tensor_output_cb_index);
            const uint32_t tile_offset = h * Wt + core_id * number_of_tiles_per_core + w;
            noc_async_write_tile(tile_offset, index_tensor_output_accessor, l1_write_addr_index);
            noc_async_write_barrier();
            cb_pop_front(index_tensor_output_cb_index, one_tile);
        }  // Wt loop

    }  // h loop
    cb_push_back(physical_core_lookup_table_cb_index, one_tile);
}
