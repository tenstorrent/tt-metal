// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/udm/mesh_builder.hpp"
#include "tt_metal/udm/mesh_utils.hpp"
#include <tt_stl/assert.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include <unordered_map>

namespace tt::tt_metal::experimental::udm {

class MeshBuilder::Impl {
public:
    Impl(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const tt::tt_metal::distributed::MeshShape& mesh_shape,
        const std::vector<tt::tt_metal::distributed::MeshCoordinate>& mesh_coords) :
        mesh_device_(mesh_device), mesh_shape_(mesh_shape), mesh_coords_(mesh_coords) {
        log_debug(
            tt::LogOp,
            "MeshBuilder::Impl constructor - mesh_shape: {}, num_coords: {}",
            mesh_shape_,
            mesh_coords_.size());

        // Create mesh object with the mesh_shape passed in as its dims
        create_mesh();
        log_debug(tt::LogOp, "  Created mesh with dims: {}", mesh_.dims);

        // Create the map: mesh -> [grids], create the grid for each mesh coordinate
        create_grids();
        if (!grids_.empty()) {
            log_debug(tt::LogOp, "  Created {} grids with shape: {}", grids_.size(), grid.dims);
            for (auto grid : grids_) {
                log_debug(tt::LogOp, "    Created grid at coord {}", grid.coord);
            }
        }

        // Create the map: [grids] -> [gcores], for each core in a grid
        create_gcores();
        log_debug(tt::LogOp, "  Created {} gcores total", gcores_.size());
        for (auto gcore : gcores_) {
            log_debug(
                tt::LogOp,
                "    Gcore: local_id={}, global_id={}, local_coord={}, global_coord={}",
                gcore.local_id,
                gcore.global_id,
                gcore.local_coord,
                gcore.global_coord);
        }
    }

    // Getters
    const Mesh& get_mesh() const { return mesh_; }

    const std::vector<Grid>& get_all_grids_in_mesh() const { return grids_; }

    const std::vector<Gcore>& get_all_gcores_in_grid(const Grid& grid) const {
        auto it = grid_to_gcores_.find(grid.id);
        TT_FATAL(it != grid_to_gcores_.end(), "Grid id {} not found", grid.id);
        return it->second;
    }

    const std::vector<Gcore>& get_all_gcores_in_mesh() const { return gcores_; }

    const Grid& get_grid_from_coord(const tt::tt_metal::distributed::MeshCoordinate& coord) const {
        auto it = coord_to_grid_.find(coord);
        TT_FATAL(it != coord_to_grid_.end(), "Mesh coordinate not found");
        return grids_[it->second];
    }

    const Gcore& get_gcore_with_local_coord(
        const tt::tt_metal::distributed::MeshCoordinate& grid_coord,
        const tt::tt_metal::distributed::MeshCoordinate& gcore_local_coord) const {
        const auto& grid = get_grid_from_coord(grid_coord);
        auto it = local_coord_to_gcore_.find({grid.id, gcore_local_coord});
        TT_FATAL(it != local_coord_to_gcore_.end(), "Local gcore coordinate not found");
        return gcores_[it->second];
    }

    const Gcore& get_gcore_with_global_coord(
        const tt::tt_metal::distributed::MeshCoordinate& gcore_global_coord) const {
        auto it = global_coord_to_gcore_.find(gcore_global_coord);
        TT_FATAL(it != global_coord_to_gcore_.end(), "Global gcore coordinate not found");
        return gcores_[it->second];
    }

    std::unordered_map<uint32_t, tt::tt_metal::CoreRangeSet> get_grid_core_range_set_from_gcores(
        const std::vector<Gcore>& gcores) const {
        std::unordered_map<uint32_t, std::vector<tt::tt_metal::CoreCoord>> grid_to_core_coords;

        // Group gcores by grid using precomputed maps
        for (const auto& gcore : gcores) {
            // find which grid this gcore belongs to
            auto grid_it = gcore_global_id_to_grid_id_.find(gcore.global_id);
            TT_FATAL(
                grid_it != gcore_global_id_to_grid_id_.end(),
                "Gcore with global_id {} not found in any grid",
                gcore.global_id);

            // get precomputed CoreCoord
            auto coord_it = gcore_global_id_to_core_coord_.find(gcore.global_id);
            TT_FATAL(
                coord_it != gcore_global_id_to_core_coord_.end(),
                "CoreCoord for gcore global_id {} not found",
                gcore.global_id);

            grid_to_core_coords[grid_it->second].push_back(coord_it->second);
        }

        // Convert vectors of CoreCoords to CoreRangeSets
        std::unordered_map<uint32_t, tt::tt_metal::CoreRangeSet> result;
        for (const auto& [grid_id, core_coords] : grid_to_core_coords) {
            result[grid_id] = tt::tt_metal::CoreRangeSet(tt::stl::Span<const tt::tt_metal::CoreCoord>(core_coords));
        }

        return result;
    }

    // Public accessor for mesh_device
    tt::tt_metal::distributed::MeshDevice* get_mesh_device() const { return mesh_device_; }

    struct MeshGridDimensions {
        uint32_t mesh_num_dims;
        std::array<uint32_t, MAX_RANK> mesh_dims;
        std::array<uint32_t, MAX_RANK> mesh_strides;
        uint32_t grid_num_dims;
        std::array<uint32_t, MAX_RANK> grid_dims;
        std::array<uint32_t, MAX_RANK> grid_strides;
    };

    MeshGridDimensions get_aligned_mesh_grid_dimensions() const {
        MeshGridDimensions result{};

        // Extract mesh dimensions
        result.mesh_num_dims = mesh_.dims.rank();
        for (uint32_t i = 0; i < result.mesh_num_dims; ++i) {
            result.mesh_dims[i] = mesh_.dims[i];
        }

        // Extract grid dimensions from first grid
        TT_FATAL(!grids_.empty(), "No grids in mesh");
        const auto& grid_shape = grids_[0].dims;
        result.grid_num_dims = grid_shape.rank();
        for (uint32_t i = 0; i < result.grid_num_dims; ++i) {
            result.grid_dims[i] = grid_shape[i];
        }

        // Align ranks for kernel args (prepend 1s for mesh, grid)
        adjust_shape_ranks(result.mesh_dims, result.mesh_num_dims, result.grid_dims, result.grid_num_dims, 1);

        // Compute strides after alignment (only once)
        compute_strides(result.mesh_dims, result.mesh_num_dims, result.mesh_strides);
        compute_strides(result.grid_dims, result.grid_num_dims, result.grid_strides);

        return result;
    }

    std::vector<uint32_t> get_compile_time_args() const {
        std::vector<uint32_t> compile_time_args;

        // MeshGcoreAccessor args layout:
        // 1. mesh_num_dims
        // 2. mesh_dims[mesh_num_dims]
        // 3. mesh_strides[mesh_num_dims]
        // 4. grid_num_dims
        // 5. grid_dims[grid_num_dims]
        // 6. grid_strides[grid_num_dims]
        // 7. num_grids
        // 8. fabric_mesh_ids[num_grids]
        // 9. fabric_chip_ids[num_grids]

        auto dims = get_aligned_mesh_grid_dimensions();

        compile_time_args.push_back(dims.mesh_num_dims);
        for (uint32_t i = 0; i < dims.mesh_num_dims; ++i) {
            compile_time_args.push_back(dims.mesh_dims[i]);
        }
        for (uint32_t i = 0; i < dims.mesh_num_dims; ++i) {
            compile_time_args.push_back(dims.mesh_strides[i]);
        }

        compile_time_args.push_back(dims.grid_num_dims);
        for (uint32_t i = 0; i < dims.grid_num_dims; ++i) {
            compile_time_args.push_back(dims.grid_dims[i]);
        }
        for (uint32_t i = 0; i < dims.grid_num_dims; ++i) {
            compile_time_args.push_back(dims.grid_strides[i]);
        }

        // Add fabric node id mapping
        compile_time_args.push_back(static_cast<uint32_t>(grids_.size()));
        for (size_t i = 0; i < grids_.size(); ++i) {
            compile_time_args.push_back(*grid_to_fabric_node_id_[i].mesh_id);
        }
        for (size_t i = 0; i < grids_.size(); ++i) {
            compile_time_args.push_back(grid_to_fabric_node_id_[i].chip_id);
        }

        return compile_time_args;
    }

    std::vector<uint32_t> get_fabric_nodes_compile_args() const {
        std::vector<uint32_t> compile_time_args;

        // Fabric node args layout:
        // 1. num_grids
        // 2. fabric_mesh_ids[num_grids]
        // 3. fabric_chip_ids[num_grids]

        compile_time_args.push_back(static_cast<uint32_t>(grids_.size()));
        for (size_t i = 0; i < grids_.size(); ++i) {
            compile_time_args.push_back(*grid_to_fabric_node_id_[i].mesh_id);
        }
        for (size_t i = 0; i < grids_.size(); ++i) {
            compile_time_args.push_back(grid_to_fabric_node_id_[i].chip_id);
        }

        return compile_time_args;
    }

private:
    tt::tt_metal::distributed::MeshCoordinate compute_gcore_local_coord(
        uint32_t flat_idx, const tt::tt_metal::Shape& grid_shape) const {
        std::vector<uint32_t> local_coord_vals(grid_shape.rank(), 0);
        uint32_t idx = flat_idx;
        for (int dim = grid_shape.rank() - 1; dim >= 0; --dim) {
            local_coord_vals[dim] = idx % grid_shape[dim];
            idx /= grid_shape[dim];
        }

        return tt::tt_metal::distributed::MeshCoordinate(
            tt::stl::Span<const uint32_t>(local_coord_vals.data(), local_coord_vals.size()));
    }

    tt::tt_metal::distributed::MeshCoordinate compute_gcore_global_coord(
        const tt::tt_metal::distributed::MeshCoordinate& grid_coord_in_mesh,
        const tt::tt_metal::Shape& grid_dims,
        const tt::tt_metal::distributed::MeshCoordinate& local_coord_in_grid) const {
        // Copy to arrays
        std::array<uint32_t, tt::tt_metal::experimental::udm::MAX_RANK> grid_coord_arr{};
        std::array<uint32_t, tt::tt_metal::experimental::udm::MAX_RANK> grid_dims_arr{};
        std::array<uint32_t, tt::tt_metal::experimental::udm::MAX_RANK> local_coord_arr{};

        uint32_t mesh_rank = grid_coord_in_mesh.dims();
        uint32_t grid_rank = grid_dims.rank();
        uint32_t local_rank = local_coord_in_grid.dims();

        for (size_t i = 0; i < mesh_rank; ++i) {
            grid_coord_arr[i] = grid_coord_in_mesh[i];
        }
        for (size_t i = 0; i < grid_rank; ++i) {
            grid_dims_arr[i] = grid_dims[i];
        }
        for (size_t i = 0; i < local_rank; ++i) {
            local_coord_arr[i] = local_coord_in_grid[i];
        }

        // Find max rank and adjust all independently to it
        uint32_t max_rank = std::max({mesh_rank, grid_rank, local_rank});

        adjust_array_to_rank(grid_coord_arr, mesh_rank, max_rank, 0);
        adjust_array_to_rank(grid_dims_arr, grid_rank, max_rank, 1);
        adjust_array_to_rank(local_coord_arr, local_rank, max_rank, 0);

        // Compute global coord: global[i] = mesh[i] * grid[i] + local[i]
        std::vector<uint32_t> global_coord_vals(max_rank);
        for (size_t i = 0; i < max_rank; ++i) {
            global_coord_vals[i] = grid_coord_arr[i] * grid_dims_arr[i] + local_coord_arr[i];
        }

        return tt::tt_metal::distributed::MeshCoordinate(
            tt::stl::Span<const uint32_t>(global_coord_vals.data(), global_coord_vals.size()));
    }

    void create_mesh() {
        // Create mesh object with the mesh_shape passed in as its dims
        std::vector<uint32_t> dims;
        for (size_t i = 0; i < mesh_shape_.dims(); ++i) {
            dims.push_back(mesh_shape_[i]);
        }
        mesh_.dims = Shape(std::move(dims));
    }

    void create_grids() {
        // Get compute grid size from mesh device
        auto compute_grid_size = mesh_device_->compute_with_storage_grid_size();
        std::vector<uint32_t> grid_dims = {compute_grid_size.y, compute_grid_size.x};

        // Reserve space for grids and fabric node mappings
        grids_.reserve(mesh_coords_.size());
        grid_to_fabric_node_id_.reserve(mesh_coords_.size());

        for (size_t i = 0; i < mesh_coords_.size(); ++i) {
            Grid grid(i, Shape(grid_dims), mesh_coords_[i]);

            grids_.push_back(grid);
            coord_to_grid_[mesh_coords_[i]] = i;

            // Extract fabric node id for this grid
            auto fabric_node_id = mesh_device_->get_fabric_node_id(mesh_coords_[i]);
            grid_to_fabric_node_id_.push_back(fabric_node_id);
        }
    }

    void create_gcores() {
        uint32_t global_gcore_id = 0;

        // For each grid, create gcores
        for (const auto& grid : grids_) {
            std::vector<Gcore> grid_gcores;
            // Get grid dimensions
            const auto& grid_shape = grid.dims;
            uint32_t grid_size = grid.size();
            grid_gcores.reserve(grid_size);

            // Iterate through all cores in the grid in row-major order
            uint32_t local_gcore_id = 0;
            for (uint32_t flat_idx = 0; flat_idx < grid_size; ++flat_idx) {
                // Build local and global coordinates
                auto local_coord = compute_gcore_local_coord(flat_idx, grid_shape);
                auto global_coord = compute_gcore_global_coord(grid.coord, grid_shape, local_coord);

                // Create gcore
                Gcore gcore(local_gcore_id, global_gcore_id, local_coord, global_coord);

                // Store gcore
                gcores_.push_back(gcore);
                grid_gcores.push_back(gcore);

                // Convert gcore to CoreCoord once at construction using safe helper
                // TODO: remove the hardcoded passing of first two coords once we support virtualizing
                //          grids, then need a mapping from [gcore.local_coord]->[core_coord]
                tt::tt_metal::CoreCoord core_coord = gcore.to_core_coord();

                // Build lookup maps
                local_coord_to_gcore_[{grid.id, gcore.local_coord}] = global_gcore_id;
                global_coord_to_gcore_[gcore.global_coord] = global_gcore_id;
                gcore_global_id_to_grid_id_[gcore.global_id] = grid.id;
                gcore_global_id_to_core_coord_[gcore.global_id] = core_coord;
                local_gcore_id++;
                global_gcore_id++;
            }

            // Store grid -> gcores mapping
            grid_to_gcores_[grid.id] = std::move(grid_gcores);
        }
    }

    // Hash function for coordinate pair in local_coord_to_gcore_ map
    struct CoordPairHash {
        size_t operator()(const std::pair<uint32_t, tt::tt_metal::distributed::MeshCoordinate>& p) const {
            return std::hash<uint32_t>{}(p.first) ^ std::hash<tt::tt_metal::distributed::MeshCoordinate>{}(p.second);
        }
    };

    tt::tt_metal::distributed::MeshDevice* mesh_device_;
    tt::tt_metal::distributed::MeshShape mesh_shape_;
    std::vector<tt::tt_metal::distributed::MeshCoordinate> mesh_coords_;

    // Constructed objects
    Mesh mesh_;
    std::vector<Grid> grids_;
    std::vector<Gcore> gcores_;

    // Lookup maps
    std::unordered_map<tt::tt_metal::distributed::MeshCoordinate, size_t> coord_to_grid_;
    std::unordered_map<uint32_t, std::vector<Gcore>> grid_to_gcores_;
    std::unordered_map<std::pair<uint32_t, tt::tt_metal::distributed::MeshCoordinate>, size_t, CoordPairHash>
        local_coord_to_gcore_;
    std::unordered_map<tt::tt_metal::distributed::MeshCoordinate, size_t> global_coord_to_gcore_;

    // Efficient mapping for get_grid_core_range_set_from_gcores: gcore global_id -> grid_id and CoreCoord
    std::unordered_map<uint32_t, uint32_t> gcore_global_id_to_grid_id_;
    std::unordered_map<uint32_t, tt::tt_metal::CoreCoord> gcore_global_id_to_core_coord_;

    // Grid to fabric node id mapping
    std::vector<tt::tt_fabric::FabricNodeId> grid_to_fabric_node_id_;
};

MeshBuilder::MeshBuilder(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    const std::vector<tt::tt_metal::distributed::MeshCoordinate>& mesh_coords) :
    impl_(std::make_unique<Impl>(mesh_device, mesh_shape, mesh_coords)) {}

MeshBuilder::MeshBuilder(const tt::tt_metal::distributed::MeshBuffer& mesh_buffer) {
    // Get mesh device from the mesh buffer
    auto* mesh_device = mesh_buffer.device();

    // Get mesh shape from the device
    auto mesh_shape = mesh_device->shape();

    // Collect all mesh coordinates in row-major order
    std::vector<tt::tt_metal::distributed::MeshCoordinate> mesh_coords;
    for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(mesh_shape)) {
        mesh_coords.push_back(coord);
    }

    // Create the impl
    impl_ = std::make_unique<Impl>(mesh_device, mesh_shape, mesh_coords);
}

MeshBuilder::~MeshBuilder() = default;

MeshBuilder::MeshBuilder(MeshBuilder&&) noexcept = default;
MeshBuilder& MeshBuilder::operator=(MeshBuilder&&) noexcept = default;

const Mesh& MeshBuilder::get_mesh() const { return impl_->get_mesh(); }

const std::vector<Grid>& MeshBuilder::get_all_grids_in_mesh() const { return impl_->get_all_grids_in_mesh(); }

const std::vector<Gcore>& MeshBuilder::get_all_gcores_in_grid(const Grid& grid) const {
    return impl_->get_all_gcores_in_grid(grid);
}

const std::vector<Gcore>& MeshBuilder::get_all_gcores_in_mesh() const { return impl_->get_all_gcores_in_mesh(); }

const Grid& MeshBuilder::get_grid_from_coord(const tt::tt_metal::distributed::MeshCoordinate& coord) const {
    return impl_->get_grid_from_coord(coord);
}

const Gcore& MeshBuilder::get_gcore_with_local_coord(
    const tt::tt_metal::distributed::MeshCoordinate& grid_coord,
    const tt::tt_metal::distributed::MeshCoordinate& gcore_coord) const {
    return impl_->get_gcore_with_local_coord(grid_coord, gcore_coord);
}

const Gcore& MeshBuilder::get_gcore_with_global_coord(
    const tt::tt_metal::distributed::MeshCoordinate& gcore_coord) const {
    return impl_->get_gcore_with_global_coord(gcore_coord);
}

std::unordered_map<uint32_t, tt::tt_metal::CoreRangeSet> MeshBuilder::get_grid_core_range_set_from_gcores(
    const std::vector<Gcore>& gcores) const {
    return impl_->get_grid_core_range_set_from_gcores(gcores);
}

tt::tt_metal::distributed::MeshDevice* MeshBuilder::mesh_device() const { return impl_->get_mesh_device(); }

std::vector<uint32_t> MeshBuilder::get_compile_time_args() const { return impl_->get_compile_time_args(); }

std::vector<uint32_t> MeshBuilder::get_fabric_nodes_compile_args() const {
    return impl_->get_fabric_nodes_compile_args();
}

}  // namespace tt::tt_metal::experimental::udm
