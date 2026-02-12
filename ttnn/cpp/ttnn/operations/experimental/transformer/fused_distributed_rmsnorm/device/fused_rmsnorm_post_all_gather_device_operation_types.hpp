// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tuple>

namespace ttnn::experimental::prim {

struct FusedRmsnormPostAllGatherParams {
    float eps;
    uint32_t num_heads;
    tt::tt_metal::MemoryConfig memory_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    std::optional<tt::tt_metal::DataType> dtype;

    static constexpr auto attribute_names =
        std::forward_as_tuple("eps", "num_heads", "memory_config", "compute_kernel_config", "dtype");
    auto attribute_values() const {
        return std::forward_as_tuple(eps, num_heads, memory_config, compute_kernel_config, dtype);
    }
};

struct FusedRmsnormPostAllGatherInputs {
    Tensor input_tensor;
    Tensor stats_tensor;
    std::optional<Tensor> weight;
    std::optional<Tensor> transformation_mat;
    std::optional<Tensor> rope_cos;
    std::optional<Tensor> rope_sin;

    static constexpr auto attribute_names =
        std::forward_as_tuple("input_tensor", "stats_tensor", "weight", "transformation_mat", "rope_cos", "rope_sin");
    auto attribute_values() const {
        return std::forward_as_tuple(input_tensor, stats_tensor, weight, transformation_mat, rope_cos, rope_sin);
    }
};

}  // namespace ttnn::experimental::prim
