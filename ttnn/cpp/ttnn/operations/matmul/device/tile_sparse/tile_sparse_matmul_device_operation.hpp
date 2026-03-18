// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/matmul/device/tile_sparse/tile_sparse_matmul_types.hpp"
#include "ttnn/operations/matmul/device/tile_sparse/factory/tile_sparse_matmul_program_factory.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"

namespace ttnn::prim {

/**
 * @brief Device operation for tile-sparse matrix multiplication.
 *
 * Performs matmul where one or both input matrices have tile-level sparsity.
 * Zero tiles (32x32 blocks of zeros) are skipped entirely during computation,
 * providing speedup proportional to tile sparsity.
 *
 * Operation: C = A @ B
 * - A can be tile-sparse with mask indicating non-zero tiles
 * - B can be tile-sparse with mask indicating non-zero tiles
 * - Output C is dense
 *
 * Implementation uses Tensix multicore with tile-based computation where:
 * - Reader kernel skips reading zero tiles based on sparsity mask
 * - Compute kernel only processes non-zero tile pairs
 * - Writer kernel writes full dense output
 */
struct TileSparseMatmulDeviceOperation {
    using operation_attributes_t = TileSparseMatmulParams;
    using tensor_args_t = TileSparseMatmulInputs;
    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    using program_factory_t = std::variant<TileSparseMatmulProgramFactory>;

    /**
     * @brief Validates operation parameters on program cache miss.
     */
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    /**
     * @brief Computes output tensor specifications.
     */
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    /**
     * @brief Creates output tensors for the operation.
     */
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    /**
     * @brief Entry point for the operation.
     *
     * @param input_tensor_a First input tensor (can be tile-sparse)
     * @param input_tensor_b Second input tensor (can be tile-sparse)
     * @param sparsity_mask_a Optional tile sparsity mask for input A
     * @param sparsity_mask_b Optional tile sparsity mask for input B
     * @param optional_output_tensor Pre-allocated output tensor
     * @param memory_config Output memory configuration
     * @param dtype Output data type
     * @param program_config Matmul program configuration
     * @param compute_kernel_config Compute kernel configuration
     * @param user_core_coord User-specified core coordinates
     * @param output_tile Output tile specification
     * @param global_cb Global circular buffer
     * @param sub_device_id Sub-device identifier
     */
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<const Tensor>& sparsity_mask_a = std::nullopt,
        const std::optional<const Tensor>& sparsity_mask_b = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<const CoreCoord>& user_core_coord = std::nullopt,
        const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
        const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);
};

/**
 * @brief Creates tile-sparse matmul attributes from input parameters.
 */
TileSparseMatmulParams create_tile_sparse_matmul_attributes(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& sparsity_mask_a,
    const std::optional<const Tensor>& sparsity_mask_b,
    const TileSparseMatmulParams& parameters,
    const std::vector<std::optional<Tensor>>& optional_output_tensors);

/**
 * @brief Creates a tile sparsity mask from a dense tensor.
 *
 * Analyzes the input tensor and creates a mask indicating which 32x32 tiles
 * contain non-zero elements (or elements above the threshold).
 *
 * @param dense_tensor Input dense tensor to analyze
 * @param threshold Minimum absolute value to consider non-zero (default: 0.0)
 * @param tile_height Height of tiles (default: 32)
 * @param tile_width Width of tiles (default: 32)
 * @return TileSparsityMask describing the tile-level sparsity pattern
 */
TileSparsityMask create_tile_sparsity_mask(
    const Tensor& dense_tensor, float threshold = 0.0f, uint32_t tile_height = 32, uint32_t tile_width = 32);

/**
 * @brief Parses a user-provided tile sparsity mask tensor.
 *
 * Converts a mask tensor (2D tensor of shape [tile_rows, tile_cols]) into
 * a TileSparsityMask struct. Non-zero values in the mask indicate non-zero tiles.
 *
 * @param mask_tensor User-provided mask tensor (must be on host)
 * @param tile_height Height of tiles (default: 32)
 * @param tile_width Width of tiles (default: 32)
 * @return TileSparsityMask with parsed tile indices
 */
TileSparsityMask parse_tile_sparsity_mask_tensor(
    const Tensor& mask_tensor, uint32_t tile_height = 32, uint32_t tile_width = 32);

// Register the operation
constexpr auto tile_sparse_matmul =
    ttnn::register_operation<"ttnn::prim::tile_sparse_matmul", TileSparseMatmulDeviceOperation>();

}  // namespace ttnn::prim
