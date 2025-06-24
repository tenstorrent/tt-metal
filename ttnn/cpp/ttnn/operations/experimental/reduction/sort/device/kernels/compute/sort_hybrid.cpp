// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_binary.h"

// #include "sort_common.hpp"

namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(0);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t number_of_tiles_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t number_of_cores_used = get_compile_time_arg_val(5);
    constexpr bool ascending = get_compile_time_arg_val(6) == 1;

    // Constants
    constexpr uint32_t one_tile = 1;
    const uint16_t core_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

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
                const uint16_t sub_dist = 1 << (sub - 1);
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

                    const bool is_ascending_block = ((tile_a >> stage) & 1) == 0;
                    const bool sort_direction = (is_ascending_block == ascending);

                    if (core_id == core_a || core_id == core_b) {
                        const uint16_t this_core = core_id;
                        const uint16_t other_core = (core_id == core_a) ? core_b : core_a;

                        bool has_a = false, has_b = false;
                        for (uint16_t i = 0; i < number_of_tiles_per_core; ++i) {
                            if (core_holds[this_core][i] == tile_a) {
                                has_a = true;
                            }
                            if (core_holds[this_core][i] == tile_b) {
                                has_b = true;
                            }
                        }  // i loop

                        // DPRINT << "  Core " << this_core << " : Compare tile " << tile_a << " with tile " << tile_b
                        //        << " — " << (sort_direction ? "ASC" : "DESC") << ENDL();

                        if (has_a && has_b) {
                            // DPRINT << "   L1-local compare of tiles " << tile_a << " and " << tile_b << ENDL();
                        } else {
                            const uint16_t local_tile = has_a ? tile_a : tile_b;
                            const uint16_t remote_tile = has_a ? tile_b : tile_a;
                            // DPRINT << "   Indirect step with core " << other_core << " for tiles: " << local_tile
                            //        << " (L1), " << remote_tile << " (remote)" << ENDL();
                        }
                    }  // if core_id == core_a || core_id == core_b
                }  // elem loop

                // DPRINT << ENDL();

            }  // sub loop
        }  // stage loop
    }  // h loop

}  // MAIN
}  // namespace NAMESPACE
