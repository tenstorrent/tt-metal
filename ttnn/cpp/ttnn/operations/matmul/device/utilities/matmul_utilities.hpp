// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::operations::matmul::utilities {

// Define the buffering depth for input CBs (0 and 1) for mcast variants.
// 2 = double buffer, 3 = triple buffer, etc.
// Allows easily changing buffering strategy in one place for relevant factories.
constexpr uint32_t MCAST_INPUT_BUFFERING_DEPTH = 2;

uint32_t get_estimated_size_of_cbs(
    uint32_t per_core_M,
    uint32_t per_core_N,
    uint32_t in0_block_w,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool transpose_a,
    bool transpose_b,
    uint32_t interm_single_tile_size,
    uint32_t bias_single_tile_size);

uint32_t estimate_interm_tile_size(
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    tt::tt_metal::DataType output_dtype);

uint32_t get_max_l1_space(const tt::tt_metal::Tensor& input_tensor_a);

bool is_input_batched(const ttnn::Shape& shape);

/**
 * @brief Computes the output shape of a matmul operation given two input tensors
 *
 * Determines the output shape based on the broadcasting rules for matrix multiplication:
 * - For 2D tensors: [m, k] @ [k, n] -> [m, n]
 * - For tensors with batch dimensions, the batch dimensions are broadcast
 * - For vector-matrix multiplication (rank 1 @ rank 2), the result is a vector
 *  Takes into account the transpose flags for the input tensors.
 *
 * @param input_tensor_a First input tensor
 * @param input_tensor_b Second input tensor
 * @param transpose_a Whether to transpose the first input tensor
 * @param transpose_b Whether to transpose the second input tensor
 * @return Shape of the resulting tensor after matmul
 */
ttnn::Shape compute_matmul_output_shape(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, bool transpose_a, bool transpose_b);

using Activation = std::variant<std::string, ttnn::operations::unary::UnaryWithParam>;
std::optional<ttnn::operations::unary::UnaryWithParam> get_fused_activation(
    const std::optional<const Activation>& activation);

tt::tt_metal::Tile get_output_tile(
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const std::optional<const tt::tt_metal::Tile>& output_tile,
    const std::optional<const tt::tt_metal::Tile>& optional_output_tensor_tile);

/**
 * @brief Calculate the M dimension for matmul operations
 *
 * @param padded_shape The padded shape of the tensor
 * @param tile The tile for the tensor (optional)
 * @param fuse_batch Whether to fuse batch dimensions
 * @return uint32_t The calculated M dimension
 */
inline uint32_t get_M_dim(
    const tt::tt_metal::Shape& padded_shape, const std::optional<tt::tt_metal::Tile>& tile, const bool fuse_batch) {
    uint32_t tile_height = tile.has_value() ? tile.value().get_height() : 1;
    if (fuse_batch) {
        return padded_shape.volume() / padded_shape[-1] / tile_height;
    }

    // Batch dims not fused, so take the height dimension
    return padded_shape[-2] / tile_height;
}

/**
 * @brief Calculate the K dimension for matmul operations
 *
 * @param padded_shape The padded shape of the tensor
 * @param tile The tile for the tensor (optional)
 * @return uint32_t The calculated K dimension
 */
inline uint32_t get_K_dim(const tt::tt_metal::Shape& padded_shape, const std::optional<tt::tt_metal::Tile>& tile) {
    return padded_shape[-1] / (tile.has_value() ? tile.value().get_width() : 1);
}

/**
 * @brief Calculate the N dimension for matmul operations
 *
 * @param padded_shape The padded shape of the tensor
 * @param tile The tile for the tensor (optional)
 * @return uint32_t The calculated N dimension
 */
inline uint32_t get_N_dim(const tt::tt_metal::Shape& padded_shape, const std::optional<tt::tt_metal::Tile>& tile) {
    return padded_shape[-1] / (tile.has_value() ? tile.value().get_width() : 1);
}

/**
 * @brief Get the padded shape of a tensor, with optional transpose.
 *
 * Returns the padded shape of the tensor. If transpose is true, the padded shape dimensions are swapped.
 *
 * @param input_tensor The tensor whose padded shape is queried.
 * @param transpose Whether to return the padded shape after transposing (swap height and width).
 * @return ttnn::Shape The padded shape of the tensor, possibly transposed.
 */
inline ttnn::Shape get_matmul_tensor_padded_shape(const Tensor& input_tensor, bool transpose) {
    auto padded_shape = input_tensor.padded_shape();
    if (transpose) {
        std::swap(padded_shape[-2], padded_shape[-1]);
    }
    return padded_shape;
}

/**
 * @brief Get the tile shape of a tensor, with optional transpose.
 *
 * Returns a tuple representing the height and width of the tensor's tile. If transpose is true,
 * the tile shape dimensions are swapped.
 *
 * @param input_tensor The tensor whose tile shape is queried.
 * @param transpose Whether to return the tile shape after transposing (swap height and width).
 * @return tt::tt_metal::Tile The tile shape of the tensor, possibly transposed.
 */
tt::tt_metal::Tile get_matmul_tile(const Tensor& input_tensor, bool transpose);

/**
 * @brief Get the shape of a tensor, with optional transpose.
 *
 * Returns the shape of the tensor. If transpose is true, the shape dimensions are swapped.
 *
 * @param input_tensor The tensor whose shape is queried.
 * @param transpose Whether to return the shape after transposing (swap height and width).
 * @return ttnn::Shape The shape of the tensor, possibly transposed.
 */
inline ttnn::Shape get_matmul_tensor_logical_shape(const Tensor& input_tensor, bool transpose) {
    auto shape = input_tensor.logical_shape();
    if (transpose) {
        std::swap(shape[-2], shape[-1]);
    }
    return shape;
}

}  // namespace ttnn::operations::matmul::utilities
