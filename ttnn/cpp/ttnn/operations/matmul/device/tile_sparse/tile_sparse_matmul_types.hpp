// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "tt-metalium/global_circular_buffer.hpp"
#include <tt_stl/reflection.hpp>

#include <vector>
#include <cstdint>

namespace ttnn::prim {

/**
 * @brief Tile Sparsity Mask for block-sparse matrix operations.
 *
 * Represents sparsity at the 32x32 tile level for Tensix-optimized computation.
 * Instead of storing individual zero elements, this tracks which tiles contain
 * non-zero data, allowing entire zero tiles to be skipped during computation.
 *
 * Storage format:
 * - mask: Boolean/uint8 tensor of shape [tile_rows, tile_cols] indicating non-zero tiles
 * - tile_indices: CSR-like index array for efficient iteration over non-zero tiles
 *
 * Example for an 8x8 tile matrix (256x256 elements):
 *   Tile mask: [1,1,0,0,1,0,0,1,
 *               1,0,0,1,0,1,0,0, ...]
 *   tile_indices: [0, 1, 4, 7, 8, 11, 13, ...]  // positions of non-zero tiles
 */
struct TileSparsityMask {
    Tensor mask;                         // bool/uint8 tensor, shape = [tile_rows, tile_cols]
    uint32_t nnz_tiles;                  // number of non-zero tiles
    std::vector<uint32_t> tile_indices;  // indices of non-zero tiles (row-major order)
    uint32_t tile_rows;                  // number of tile rows
    uint32_t tile_cols;                  // number of tile columns

    TileSparsityMask() : nnz_tiles(0), tile_rows(0), tile_cols(0) {}

    TileSparsityMask(Tensor mask_tensor, uint32_t nnz, std::vector<uint32_t> indices, uint32_t rows, uint32_t cols) :
        mask(std::move(mask_tensor)),
        nnz_tiles(nnz),
        tile_indices(std::move(indices)),
        tile_rows(rows),
        tile_cols(cols) {}

    // Calculate sparsity ratio (fraction of zero tiles)
    float sparsity_ratio() const {
        if (tile_rows == 0 || tile_cols == 0) {
            return 0.0f;
        }
        uint32_t total_tiles = tile_rows * tile_cols;
        return 1.0f - (static_cast<float>(nnz_tiles) / static_cast<float>(total_tiles));
    }

    // Check if dense (no sparsity)
    bool is_dense() const { return nnz_tiles == tile_rows * tile_cols; }

    // Custom hash function for operation caching
    // Only hashes scalar metadata, not the tensor or index vector
    tt::stl::hash::hash_t to_hash() const {
        return tt::stl::hash::hash_objects_with_default_seed(nnz_tiles, tile_rows, tile_cols);
    }
};

// Stream operator for TileSparsityMask (required for reflection/logging)
inline std::ostream& operator<<(std::ostream& os, const TileSparsityMask& mask) {
    os << "TileSparsityMask{tile_rows=" << mask.tile_rows << ", tile_cols=" << mask.tile_cols
       << ", nnz_tiles=" << mask.nnz_tiles << ", sparsity=" << mask.sparsity_ratio() << "}";
    return os;
}

/**
 * @brief Parameters for tile-sparse matrix multiplication.
 *
 * Extends standard matmul parameters with tile sparsity mask information.
 */
struct TileSparseMatmulParams {
    // Tile sparsity configuration
    std::optional<TileSparsityMask> input_a_sparsity_mask;  // Sparsity mask for input A
    std::optional<TileSparsityMask> input_b_sparsity_mask;  // Sparsity mask for input B

    // Standard matmul parameters
    std::optional<const operations::matmul::MatmulProgramConfig> program_config = std::nullopt;
    tt::tt_metal::MemoryConfig output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    std::optional<tt::tt_metal::DataType> output_dtype = std::nullopt;
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
    std::optional<const CoreCoord> user_core_coord = std::nullopt;
    std::optional<const tt::tt_metal::Tile> output_tile;
    std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;

    // Tile size (default: 32x32 for Tensix architecture)
    uint32_t tile_height = 32;
    uint32_t tile_width = 32;

    // Custom hash function for operation caching
    tt::stl::hash::hash_t to_hash() const {
        // Hash the sparsity mask metadata (if present) and standard matmul params
        tt::stl::hash::hash_t mask_a_hash = input_a_sparsity_mask.has_value() ? input_a_sparsity_mask->to_hash() : 0;
        tt::stl::hash::hash_t mask_b_hash = input_b_sparsity_mask.has_value() ? input_b_sparsity_mask->to_hash() : 0;
        return tt::stl::hash::hash_objects_with_default_seed(
            mask_a_hash,
            mask_b_hash,
            program_config,
            output_mem_config,
            output_dtype,
            compute_kernel_config,
            user_core_coord,
            output_tile,
            global_cb,
            sub_device_id,
            tile_height,
            tile_width);
    }
};

/**
 * @brief Input tensors for tile-sparse matmul operation.
 */
struct TileSparseMatmulInputs {
    std::vector<Tensor> input_tensors;                                // [input_a, input_b]
    std::vector<std::optional<const Tensor>> optional_input_tensors;  // [sparsity_mask_a, sparsity_mask_b]
    std::vector<std::optional<Tensor>> optional_output_tensors;
};

}  // namespace ttnn::prim
