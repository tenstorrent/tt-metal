// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

using Activation = std::variant<std::string, UnaryWithParam>;

namespace operations::matmul {

namespace detail {

bool is_input_batched(const ttnn::Shape& logical_shape);

}  // namespace detail

std::optional<UnaryWithParam> get_fused_activation(const std::optional<const Activation>& activation);

Tensor matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool transpose_a = false,
    bool transpose_b = false,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const MatmulProgramConfig>& program_config = std::nullopt,
    const std::optional<const Activation>& activation = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<const CoreGrid> core_grid = std::nullopt,
    const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
    std::optional<Tensor> optional_output_tensor = std::nullopt,
    const std::optional<const GlobalCircularBuffer>& global_cb = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);

std::vector<Tensor> matmul_batched_weights(
    const Tensor& input_tensor_a,
    const std::vector<Tensor>& input_tensors_b,
    bool transpose_a = false,
    bool transpose_b = false,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const MatmulProgramConfig>& program_config = std::nullopt,
    const std::optional<const Activation>& activation = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<const CoreGrid> core_grid = std::nullopt,
    const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<const GlobalCircularBuffer>& global_cb = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);

Tensor linear(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& bias = std::nullopt,
    bool transpose_a = false,
    bool transpose_b = false,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const MatmulProgramConfig>& program_config = std::nullopt,
    const std::optional<const Activation>& activation = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<const CoreGrid> core_grid = std::nullopt,
    const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
    std::optional<Tensor> optional_output_tensor = std::nullopt,
    const std::optional<const GlobalCircularBuffer>& global_cb = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);

void addmm_validate(
    const Tensor& input_tensor, const Tensor& mat1_tensor, const Tensor& mat2_tensor, float alpha, float beta);

Tensor addmm(
    const Tensor& input_tensor,
    const Tensor& mat1_tensor,
    const Tensor& mat2_tensor,
    float alpha = 1.0f,
    float beta = 1.0f,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const MatmulProgramConfig>& program_config = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<const CoreGrid> core_grid = std::nullopt,
    const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
    std::optional<Tensor> optional_output_tensor = std::nullopt);

Tensor sparse_matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Tensor& sparsity,
    const MatmulProgramConfig& program_config,
    std::optional<uint32_t> nnz = std::nullopt,
    bool is_input_a_sparse = false,
    bool is_input_b_sparse = true,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<const CoreGrid> core_grid = std::nullopt,
    const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<const GlobalCircularBuffer>& global_cb = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);

/**
 * @brief Tile-sparse matrix multiplication.
 *
 * Performs matmul where one or both input matrices have tile-level sparsity.
 * Zero tiles (32x32 blocks of zeros) are skipped during computation,
 * providing speedup proportional to tile sparsity.
 *
 * @param input_tensor_a First input tensor (can have tile sparsity)
 * @param input_tensor_b Second input tensor (can have tile sparsity)
 * @param sparsity_mask_a Optional tile sparsity mask for input A (uint8 tensor of shape [tile_rows_a, tile_cols_a])
 * @param sparsity_mask_b Optional tile sparsity mask for input B (uint8 tensor of shape [tile_rows_b, tile_cols_b])
 * @param memory_config Output memory configuration
 * @param dtype Output data type
 * @param program_config Matmul program configuration
 * @param compute_kernel_config Compute kernel configuration
 * @param core_grid Core grid for computation
 * @param output_tile Output tile specification
 * @param optional_output_tensor Pre-allocated output tensor
 * @param global_cb Global circular buffer
 * @param sub_device_id Sub-device identifier
 * @return Output tensor C = A @ B
 */
Tensor tile_sparse_matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& sparsity_mask_a = std::nullopt,
    const std::optional<const Tensor>& sparsity_mask_b = std::nullopt,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const MatmulProgramConfig>& program_config = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<const CoreGrid> core_grid = std::nullopt,
    const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<const GlobalCircularBuffer>& global_cb = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);

/**
 * @brief Create a tile sparsity mask from a dense tensor.
 *
 * Analyzes the input tensor and creates a boolean mask indicating which 32x32 tiles
 * contain non-zero elements (or elements above the threshold).
 *
 * @param dense_tensor Input dense tensor to analyze
 * @param threshold Minimum absolute value to consider non-zero (default: 0.0)
 * @return Tensor of shape [tile_rows, tile_cols] with uint8 values (0 or 1)
 */
Tensor create_tile_sparsity_mask(const Tensor& dense_tensor, float threshold = 0.0f);

}  // namespace operations::matmul

// Export to ttnn namespace
using operations::matmul::addmm;
using operations::matmul::create_tile_sparsity_mask;
using operations::matmul::linear;
using operations::matmul::matmul;
using operations::matmul::matmul_batched_weights;
using operations::matmul::sparse_matmul;
using operations::matmul::tile_sparse_matmul;

}  // namespace ttnn
