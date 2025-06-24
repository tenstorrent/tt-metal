// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

#include <array>
#include <algorithm>
#include <cstdint>

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
    constexpr uint32_t physical_core_lookup_table_cb_index = get_compile_time_arg_val(4);
    constexpr bool input_tensor_is_dram = get_compile_time_arg_val(5) == 1;
    constexpr bool index_tensor_output_is_dram = get_compile_time_arg_val(6) == 1;
    constexpr bool physical_core_lookup_table_is_dram = get_compile_time_arg_val(7) == 1;
    constexpr uint32_t Ht = get_compile_time_arg_val(8);
    constexpr uint32_t Wt = get_compile_time_arg_val(9);
    constexpr uint32_t number_of_tiles_per_core = get_compile_time_arg_val(10);
    constexpr uint32_t number_of_cores_used = get_compile_time_arg_val(11);
    constexpr bool ascending = get_compile_time_arg_val(12) == 1;

    // Constants
    constexpr uint32_t one_tile = 1;
    const uint16_t core_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

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
    /*
        for (uint32_t h = 0; h < Ht; h++) {
            // Tiles for each core placeholder
            std::array<std::array<uint16_t, number_of_tiles_per_core>, number_of_cores_used> core_holds{};
            for (uint16_t core = 0; core < number_of_cores_used; ++core) {
                for (uint16_t i = 0; i < number_of_tiles_per_core; ++i) {
                    core_holds[core][i] = core * number_of_tiles_per_core + i;
                }
            }

            // Calculate number of stages for bitonic sort
            uint16_t stages = 0;
            for (uint16_t i = Wt; i > 1; i >>= 1) {
                stages++;
            }
            // Sort tiles for each core
            for (uint16_t stage = 1; stage <= stages; ++stage) {
                // DPRINT << "Stage " << stage << ":" << ENDL();
                for (uint16_t sub = stage; sub > 0; --sub) {
                    uint16_t sub_dist = 1 << (sub - 1);
                    // DPRINT << " Sub-stage " << sub << " (compare distance = " << sub_dist << "):" << ENDL();

                    std::array<bool, Wt> processed{};
                    for (uint16_t elem = 0; elem < Wt; ++elem) {
                        if (processed[elem]) {
                            continue;
                        }

                        uint16_t partner = elem ^ sub_dist;
                        if (partner >= Wt) {
                            continue;
                        }

                        processed[elem] = processed[partner] = true;
                        const uint16_t tile_a = std::min(elem, partner);
                        const uint16_t tile_b = std::max(elem, partner);

                        // Lookup cores that hold tile_a and tile_b
                        int core_a = -1, core_b = -1;
                        for (uint16_t core = 0; core < number_of_cores_used; ++core) {
                            for (uint16_t i = 0; i < number_of_tiles_per_core; ++i) {
                                if (core_holds[core][i] == tile_a) {
                                    core_a = core;
                                }
                                if (core_holds[core][i] == tile_b) {
                                    core_b = core;
                                }
                            }  // i loop
                        }  // core loop

                        bool is_ascending_block = ((tile_a >> stage) & 1) == 0;
                        bool sort_direction = (is_ascending_block == ascending);

                        if (core_id == core_a || core_id == core_b) {
                            uint16_t this_core = core_id;
                            uint16_t other_core = (core_id == core_a) ? core_b : core_a;

                            bool has_a = false, has_b = false;
                            for (uint16_t i = 0; i < number_of_tiles_per_core; ++i) {
                                if (core_holds[this_core][i] == tile_a) {
                                    has_a = true;
                                }
                                if (core_holds[this_core][i] == tile_b) {
                                    has_b = true;
                                }
                            }  // i loop

                            // DPRINT << "  Core " << this_core << " : Compare tile " << tile_a << " with tile " <<
       tile_b
                            //        << " — " << (sort_direction ? "ASC" : "DESC") << ENDL();

                            if (has_a && has_b) {
                                // DPRINT << "   L1-local compare of tiles " << tile_a << " and " << tile_b << ENDL();
                            } else {
                                uint16_t local_tile = has_a ? tile_a : tile_b;
                                uint16_t remote_tile = has_a ? tile_b : tile_a;
                                // DPRINT << "   Indirect step with core " << other_core << " for tiles: " << local_tile
                                    //    << " (L1), " << remote_tile << " (remote)" << ENDL();
                            }
                        }  // if core_id == core_a || core_id == core_b
                    }  // elem loop

                    DPRINT << ENDL();

                }  // sub loop
            }  // stage loop
        }  // h loop
    */
}
