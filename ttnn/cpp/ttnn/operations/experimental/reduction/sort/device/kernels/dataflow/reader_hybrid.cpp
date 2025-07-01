// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

#include "hybrid_common.hpp"

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

    DPRINT << "READER: "
           << " grid_x: " << compute_with_storage_grid_size_x << " grid_y: " << compute_with_storage_grid_size_y
           << " input_cb_index: " << input_tensor_cb_index << " index_output_cb_index: " << index_tensor_output_cb_index
           << " value_intermediate = " << value_tensor_intermediate_cb_index
           << " index_intermediate = " << index_tensor_intermediate_cb_index
           << " value_peer_cb_index: " << value_tensor_peer_cb_index
           << " index_peer_cb_index: " << index_tensor_peer_cb_index
           << " physical_core_lookup_table_cb_index: " << physical_core_lookup_table_cb_index << " Ht: " << Ht
           << " Wt: " << Wt << " number_of_tiles_per_core: " << number_of_tiles_per_core
           << " number_of_cores_used: " << number_of_cores_used << " ascending: " << (uint32_t)ascending
           << " sem_exchange_addr: " << sem_exchange_addr << ENDL();

    // Constants
    constexpr uint32_t one_tile = 1;
    const uint16_t core_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();
    const uint16_t global_tile_start = core_id * number_of_tiles_per_core;
    const uint16_t global_tile_end = global_tile_start + number_of_tiles_per_core;

    // const uint16_t number_of_pairs_processed_by_each_core = number_of_tiles_per_core / 2;
    const uint16_t processing_tile_start = core_id * number_of_tiles_per_core;
    const uint16_t processing_tile_end = processing_tile_start + number_of_tiles_per_core;

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

    DPRINT << "READER: Generating core LUT" << ENDL();
    // Read lookup table for physical core IDs
    cb_reserve_back(physical_core_lookup_table_cb_index, one_tile);
    const uint32_t physical_core_lookup_table_l1_write_addr = get_write_ptr(physical_core_lookup_table_cb_index);
    uint64_t noc_addr = get_noc_addr(0, physical_core_lookup_table_accessor);
    noc_async_read(noc_addr, physical_core_lookup_table_l1_write_addr, physical_core_lookup_table_tile_size_bytes);
    noc_async_read_barrier();

    sem_ptr_t sem_self_exchange_ptr = reinterpret_cast<sem_ptr_t>(sem_exchange_addr);

    DPRINT << "READER: Starting" << ENDL();  // TODO: Remove

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

        DPRINT << "READER: LOGIC" << ENDL();
        uint32_t stages = ilog2(Wt);
        DPRINT << "READER: stages = " << stages << ENDL();
        for (uint32_t stage = 2; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);
                uint16_t pair_id = 0;

                // for (uint32_t i = 0; i < Wt; i++) {
                //     uint32_t j = i ^ sub_dist;
                //     if (j > i) {
                //         if (pair_id >= processing_pair_start && pair_id < processing_pair_end) {
                //             if (i >= global_tile_start && i < global_tile_end && j >= global_tile_start &&
                //                 j < global_tile_end) {
                //                 // NOTHING
                //             } else {
                //                 // TODO: Swapping tiles
                //                 // Get second core id
                //                 const uint32_t other_core_id = j / number_of_tiles_per_core;
                //                 const std::pair<uint32_t, uint32_t> remote_core_physical =
                //                     get_core_physical_coordinates(other_core_id,
                //                     physical_core_lookup_table_cb_index);
                //             }
                //         }
                //         pair_id++;
                //     }
                // }

                // TOOD: We don't need to check for each tile if it's outside or inside core. We can simply check the
                // first one
                //       For a given sub, all tiles are either in-core or outside
                //       If inside => do nothing
                //      Otherwise => exchange

                uint32_t i = processing_tile_start;
                uint32_t j = i ^ sub_dist;
                DPRINT << "READER: i = " << i << ", j = " << j << ", processing pair start = " << processing_tile_start
                       << ", pair end = " << processing_tile_end << ENDL();
                // if (pair_id >= processing_tile_start && pair_id < processing_tile_end) {
                if (i >= global_tile_start && i < global_tile_end && j >= global_tile_start && j < global_tile_end) {
                    // Nothing
                } else {
                    const uint32_t other_core_id = j / number_of_tiles_per_core;
                    const std::pair<uint32_t, uint32_t> remote_core_physical =
                        get_core_physical_coordinates(other_core_id, physical_core_lookup_table_cb_index);

                    DPRINT << "READER: Exchanging tile with " << other_core_id << " (" << remote_core_physical.first
                           << ", " << remote_core_physical.second << ")" << ENDL();

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

                    DPRINT << "READER: Tiles have been exchanged" << ENDL();
                }
                // }
            }  // sub

            // TODO: PUT BARRIER HERE

        }  // stages

        DPRINT << "READER: AFTER LOGIC:" << ENDL();  // TODO: Remove
        // Write output index data
        for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
            DPRINT << "WRITER: Writing tile: " << w << " at h: " << h << ENDL();  // TODO: remove
            cb_wait_front(index_tensor_output_cb_index, one_tile);
            const uint32_t l1_write_addr_index = get_read_ptr(index_tensor_output_cb_index);
            const uint32_t tile_offset = h * Wt + core_id * number_of_tiles_per_core + w;
            noc_async_write_tile(tile_offset, index_tensor_output_accessor, l1_write_addr_index);
            noc_async_write_barrier();
            cb_pop_front(index_tensor_output_cb_index, one_tile);
        }  // Wt loop

    }  // h loop
    cb_push_back(physical_core_lookup_table_cb_index, one_tile);

    DPRINT << "READER: Finished reading and sorting tiles." << ENDL();  // TODO: Remove
}
