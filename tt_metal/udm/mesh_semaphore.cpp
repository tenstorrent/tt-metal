// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/udm/mesh_semaphore.hpp"
#include "tt_metal/udm/mesh_builder.hpp"
#include "tt_metal/udm/mesh_program.hpp"
#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/distributed.hpp"
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::experimental::udm {

MeshSemaphoreHandle CreateMeshSemaphore(MeshBuilder& builder, MeshProgram& program, uint32_t initial_value) {
    // Get all gcores from mesh builder
    const auto& all_gcores_vec = builder.get_all_gcores_in_mesh();
    // Get grid IDs and local CoreRangeSets from gcores
    auto grid_to_cores = builder.get_grid_core_range_set_from_gcores(all_gcores_vec);

    MeshSemaphoreHandle mesh_semaphore_handle;

    // Create semaphore on each grid
    for (const auto& [grid_id, core_range_set] : grid_to_cores) {
        // Get the grid object to access its mesh coordinate
        const auto& all_grids = builder.get_all_grids_in_mesh();
        const Grid* grid_ptr = nullptr;
        for (const auto& g : all_grids) {
            if (g.id == grid_id) {
                grid_ptr = &g;
                break;
            }
        }
        TT_FATAL(grid_ptr != nullptr, "Grid {} not found", grid_id);

        auto& grid_program = program.program_at(grid_ptr->coord);

        auto address = tt::tt_metal::CreateSemaphore(grid_program, core_range_set, initial_value);

        // Store the semaphore address for this grid
        mesh_semaphore_handle[grid_id] = address;
    }

    // Synchronize mesh device after creating semaphores on all grids
    auto* mesh_device = builder.mesh_device();
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});

    return mesh_semaphore_handle;
}

}  // namespace tt::tt_metal::experimental::udm
