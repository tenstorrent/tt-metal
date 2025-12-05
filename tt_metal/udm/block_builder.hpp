// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <unordered_map>
#include "tt_metal/udm/types.hpp"
#include "tt_metal/common/core_coord.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::udm {

/**
 * @brief Builder class for constructing Block abstractions from distributed tensors
 *
 * BlockBuilder creates the necessary mappings:
 * - [gcore_id] -> [device_bank_id]
 * - [grid_index] -> [gcore_id]
 * - [block_index] -> [fabric_node_id]
 */
class BlockBuilder {
public:
    /**
     * @brief Construct a BlockBuilder from mesh grid information
     *
     * @param mesh_device The mesh grid
     * @param mesh_shape The shape of the mesh
     * @param mesh_coords The coordinates in the mesh
     * @param grid Grid per grid
     * @param block Block shape
     */
    BlockBuilder(
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const tt::tt_metal::MeshShape& mesh_shape,
        const std::vector<ttnn::MeshCoordinate>& mesh_coords,
        const Grid& grid,
        const Block& block);

    ~BlockBuilder();

    // Delete copy, allow move
    BlockBuilder(const BlockBuilder&) = delete;
    BlockBuilder& operator=(const BlockBuilder&) = delete;
    BlockBuilder(BlockBuilder&&) noexcept;
    BlockBuilder& operator=(BlockBuilder&&) noexcept;

    /**
     * @brief Get the Block configuration
     */
    const Block& block() const;

    /**
     * @brief Map gcore_id to device_bank_id
     */
    uint32_t gcore_to_device_bank_id(const Gcore& gcore) const;

    /**
     * @brief Map grid_index to gcore_id
     */
    Gcore grid_index_to_gcore(uint32_t grid_index) const;

    /**
     * @brief Map block_index to fabric_node_id
     */
    tt::tt_fabric::FabricNodeId block_index_to_fabric_node_id(uint32_t block_index) const;

    /**
     * @brief Get the mesh grid from the input tensor
     */
    tt::tt_metal::distributed::MeshDevice* mesh_device() const;

private:
    friend class TensorBuilder;  // Only TensorBuilder can create BlockBuilder
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::udm
