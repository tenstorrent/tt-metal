// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/matmul/device/tmp/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "tt-metalium/global_circular_buffer.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace ttnn::operations::matmul::config;  // TODO:[migration] remove after migration
namespace ttnn::operations::matmul::tmp {          // TODO:[migration] remove after migration

enum class Matmul1DType { MCAST_IN0, GATHER_IN0, MCAST_IN1 };
}

namespace ttnn::operations::matmul {

// from struct Matrix which is used but run
struct operation_attributes_t {
    std::optional<MatmulProgramConfig> program_config;
    std::optional<bool> bcast_batch;  // TODO: why optional? can we make it required?
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config;
    bool untilize_out;
    std::optional<CoreCoord> user_core_coord;
    std::optional<ttnn::operations::unary::UnaryWithParam> user_fused_activation;
    bool user_run_batched;
    bool transpose_a;
    bool transpose_b;
    std::optional<tt::tt_metal::Tile> output_tile;
    std::optional<tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
};

// std::vector<Tensor> input_tensors;
// std::vector<std::optional<Tensor>> optional_input_tensors;
// std::vector<std::optional<Tensor>> output_tensors;
struct tensor_args_t {
    Tensor input_tensor_a;
    Tensor input_tensor_b;
    std::optional<Tensor> bias;
    std::optional<Tensor> output_tensor;
};

using spec_return_value_t = ttnn::TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::matmul

namespace ttnn::operations::matmul_sparse {

struct operation_attributes_t {
    std::optional<uint32_t> nnz;
    bool is_input_a_sparse;
    bool is_input_b_sparse;
    std::optional<const MatmulProgramConfig> program_config = std::nullopt;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype = std::nullopt;
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
    std::optional<const CoreCoord> user_core_coord = std::nullopt;
    std::optional<const tt::tt_metal::Tile> output_tile;
    std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
};

struct tensor_args_t {
    Tensor input_tensor_a;
    Tensor input_tensor_b;
    Tensor sparsity;
    std::optional<Tensor> bias;  // For spare?
};

using spec_return_value_t = std::vector<ttnn::TensorSpec>;  // TODO: Valdiate
using tensor_return_value_t = std::vector<Tensor>;          // TODO: Valdiate
}  // namespace ttnn::operations::matmul_sparse