// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/udm/block_builder.hpp"
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::udm {

class BlockBuilder::Impl {
public:
    Impl(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const tt::tt_metal::MeshShape& mesh_shape,
        const std::vector<ttnn::MeshCoordinate>& mesh_coords,
        const Grid& grid,
        const Block& block) :
        mesh_device_(mesh_device),
        mesh_shape_(mesh_shape),
        mesh_coords_(mesh_coords),
        grid_template_(grid),
        block_template_(block) {
        // Build Block and Grid structures
        build_block_and_grids();
        // TODO: Implement mapping logic
    }

    void build_block_and_grids() {
        // 1. Determine block dimensions
        std::vector<uint32_t> block_dims;
        if (block_template_.dimensions.empty()) {
            // Use actual mesh shape as block shape
            for (size_t i = 0; i < mesh_shape_.dims(); ++i) {
                block_dims.push_back(mesh_shape_[i]);
            }
        } else {
            block_dims = block_template_.dimensions;
        }
        block_.dimensions = block_dims;

        // 2. Determine grid dimensions
        std::vector<uint32_t> grid_dims;
        if (grid_template_.dimensions.empty()) {
            auto compute_grid = mesh_device_->compute_with_storage_grid_size();
            grid_dims = {compute_grid.y, compute_grid.x};  // rows, cols
        } else {
            grid_dims = grid_template_.dimensions;
        }

        // 3. Create grids for each grid
        size_t num_devices = mesh_coords_.size();
        block_.grids.resize(num_devices);

        uint32_t global_gcore_id = 0;
        for (size_t grid_idx = 0; grid_idx < num_devices; ++grid_idx) {
            Grid& grid = block_.grids[grid_idx];
            grid.dimensions = grid_dims;

            // Create gcores for this grid
            uint32_t num_cores = 1;
            for (auto dim : grid_dims) {
                num_cores *= dim;
            }

            for (uint32_t i = 0; i < num_cores; ++i) {
                Gcore gcore;
                gcore.id = global_gcore_id++;
                grid.gcores.push_back(gcore);
            }
        }

        // 4. Build mappings (TODO: implement the actual mapping logic)
        // - gcore_to_device_bank_map_
        // - grid_to_gcore_map_
        // - block_to_fabric_node_map_
    }

    const Block& block() const { return block_; }

    uint32_t gcore_to_device_bank_id(const Gcore& gcore) const {
        // TODO: Implement mapping
        TT_FATAL(false, "gcore_to_device_bank_id not yet implemented");
        return 0;
    }

    Gcore grid_index_to_gcore(uint32_t grid_index) const {
        // TODO: Implement mapping
        TT_FATAL(false, "grid_index_to_gcore not yet implemented");
        return Gcore{0};
    }

    tt::tt_fabric::FabricNodeId block_index_to_fabric_node_id(uint32_t block_index) const {
        // TODO: Implement mapping
        TT_FATAL(false, "block_index_to_fabric_node_id not yet implemented");
        return tt::tt_fabric::FabricNodeId{};
    }

    tt::tt_metal::distributed::MeshDevice* mesh_device() const { return mesh_device_; }

private:
    tt::tt_metal::distributed::MeshDevice* mesh_device_;
    tt::tt_metal::MeshShape mesh_shape_;
    std::vector<ttnn::MeshCoordinate> mesh_coords_;
    Grid grid_template_;    // Template for grid creation
    Block block_template_;  // Template for block creation

    Block block_;  // The actual built block

    // Mappings
    std::unordered_map<uint32_t, uint32_t> gcore_to_device_bank_map_;
    std::unordered_map<uint32_t, Gcore> grid_to_gcore_map_;
    std::unordered_map<uint32_t, tt::tt_fabric::FabricNodeId> block_to_fabric_node_map_;
};

BlockBuilder::BlockBuilder(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const tt::tt_metal::MeshShape& mesh_shape,
    const std::vector<ttnn::MeshCoordinate>& mesh_coords,
    const Grid& grid,
    const Block& block) :
    impl_(std::make_unique<Impl>(mesh_device, mesh_shape, mesh_coords, grid, block)) {}

BlockBuilder::~BlockBuilder() = default;

BlockBuilder::BlockBuilder(BlockBuilder&&) noexcept = default;
BlockBuilder& BlockBuilder::operator=(BlockBuilder&&) noexcept = default;

const Block& BlockBuilder::block() const { return impl_->block(); }

uint32_t BlockBuilder::gcore_to_device_bank_id(const Gcore& gcore) const {
    return impl_->gcore_to_device_bank_id(gcore);
}

Gcore BlockBuilder::grid_index_to_gcore(uint32_t grid_index) const { return impl_->grid_index_to_gcore(grid_index); }

tt::tt_fabric::FabricNodeId BlockBuilder::block_index_to_fabric_node_id(uint32_t block_index) const {
    return impl_->block_index_to_fabric_node_id(block_index);
}

tt::tt_metal::distributed::MeshDevice* BlockBuilder::mesh_device() const { return impl_->mesh_device(); }

}  // namespace tt::tt_metal::udm
