// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

namespace ttnn::experimental::prim {

struct MinimalMatmulSplitParams {
    std::optional<MinimalMatmulConfig> config;
    std::optional<operations::unary::UnaryWithParam> fused_activation;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
    int32_t chunks{};
    int32_t dim{};

    static constexpr auto attribute_names = std::forward_as_tuple(
        "config", "fused_activation", "output_mem_config", "output_dtype", "compute_kernel_config", "chunks", "dim");
    auto attribute_values() const {
        return std::forward_as_tuple(
            config, fused_activation, output_mem_config, output_dtype, compute_kernel_config, chunks, dim);
    }
};

struct MinimalMatmulSplitInputs {
    Tensor input_tensor;
    Tensor weight_tensor;
    std::optional<Tensor> bias_tensor;

    static constexpr auto attribute_names = std::forward_as_tuple("input_tensor", "weight_tensor", "bias_tensor");
    auto attribute_values() const { return std::forward_as_tuple(input_tensor, weight_tensor, bias_tensor); }
};

}  // namespace ttnn::experimental::prim
