// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

Tensor matmul_full_grid_precise(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, bool transpose_a = false, bool transpose_b = false);

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

}  // namespace operations::matmul

// Export to ttnn namespace
using operations::matmul::addmm;
using operations::matmul::linear;
using operations::matmul::matmul;
using operations::matmul::matmul_batched_weights;
using operations::matmul::sparse_matmul;

}  // namespace ttnn
