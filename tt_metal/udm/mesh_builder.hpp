// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <set>
#include <unordered_map>
#include <vector>
#include "tt_metal/udm/types.hpp"
#include "tt_metal/common/core_coord.hpp"
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>

namespace tt::tt_metal::experimental::udm {

/**
 * @brief Builder class for constructing abstraction of Mesh/Grid/Gcore
 *
 */
class MeshBuilder {
public:
    /**
     * @brief Construct a MeshBuilder from mesh grid information
     *
     * @param mesh_device The mesh grid
     * @param mesh_shape The shape of the mesh
     * @param mesh_coords The coordinates in the mesh
     */
    MeshBuilder(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const tt::tt_metal::distributed::MeshShape& mesh_shape,
        const std::vector<tt::tt_metal::distributed::MeshCoordinate>& mesh_coords);

    /**
     * @brief Construct a MeshBuilder from a MeshBuffer
     *
     * @param mesh_buffer The mesh buffer containing device information
     */
    explicit MeshBuilder(const tt::tt_metal::distributed::MeshBuffer& mesh_buffer);

    ~MeshBuilder();

    // Delete copy, allow move
    MeshBuilder(const MeshBuilder&) = delete;
    MeshBuilder& operator=(const MeshBuilder&) = delete;
    MeshBuilder(MeshBuilder&&) noexcept;
    MeshBuilder& operator=(MeshBuilder&&) noexcept;

    // Getters
    const Mesh& get_mesh() const;

    const std::vector<Grid>& get_all_grids_in_mesh() const;

    const std::vector<Gcore>& get_all_gcores_in_grid(const Grid& grid) const;

    const std::vector<Gcore>& get_all_gcores_in_mesh() const;

    const Grid& get_grid_from_coord(const tt::tt_metal::distributed::MeshCoordinate& coord) const;

    const Gcore& get_gcore_with_local_coord(
        const tt::tt_metal::distributed::MeshCoordinate& grid_coord,
        const tt::tt_metal::distributed::MeshCoordinate& gcore_coord) const;

    const Gcore& get_gcore_with_global_coord(const tt::tt_metal::distributed::MeshCoordinate& gcore_coord) const;

    /**
     * @brief Get grid IDs and CoreRangeSets from global cores
     *
     * @param gcores The global cores to map
     * @return Map from grid_id to CoreRangeSet of local coordinates
     */
    std::unordered_map<uint32_t, tt::tt_metal::CoreRangeSet> get_grid_core_range_set_from_gcores(
        const std::vector<Gcore>& gcores) const;

    tt::tt_metal::distributed::MeshDevice* mesh_device() const;

    /**
     * @brief Get compile-time arguments for MeshGcoreAccessor
     *
     * @return Vector of compile-time arguments containing mesh and grid topology
     */
    std::vector<uint32_t> get_compile_time_args() const;

    /**
     * @brief Get fabric node compile-time arguments
     *
     * @return Vector of compile-time arguments containing:
     *         1. num_grids
     *         2. fabric_mesh_ids[num_grids]
     *         3. fabric_chip_ids[num_grids]
     */
    std::vector<uint32_t> get_fabric_nodes_compile_args() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::experimental::udm
