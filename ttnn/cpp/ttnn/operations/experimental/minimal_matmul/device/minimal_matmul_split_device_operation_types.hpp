// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"

namespace ttnn::operations::experimental::minimal_matmul {

struct split_operation_attributes_t {
    std::optional<MinimalMatmulConfig> config;
    std::optional<unary::UnaryWithParam> fused_activation;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
    int32_t chunks;
    int32_t dim;
};

struct split_tensor_args_t {
    Tensor input_tensor;
    Tensor weight_tensor;
    std::optional<Tensor> bias_tensor;
};

using split_spec_return_value_t = std::vector<TensorSpec>;
using split_tensor_return_value_t = std::vector<Tensor>;

}  // namespace ttnn::operations::experimental::minimal_matmul
