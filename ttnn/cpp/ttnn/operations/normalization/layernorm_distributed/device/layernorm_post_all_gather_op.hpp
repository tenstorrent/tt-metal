// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

#include "layernorm_distributed_types.hpp"

using namespace tt::constants;

namespace ttnn::operations::normalization {

tt::tt_metal::operation::ProgramWithCallbacks layernorm_post_allgather_multi_core(
    const Tensor& a,
    const Tensor& stats,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    Tensor& output,
    LayerNormDistributedType norm_type,
    float eps,
    DeviceComputeKernelConfig compute_kernel_config,
    std::optional<bool> use_2d_core_grid = std::nullopt);

struct LayerNormPostAllGather {
    LayerNormDistributedType norm_type;
    float eps;
    MemoryConfig memory_config;
    const DeviceComputeKernelConfig compute_kernel_config;
    std::optional<DataType> dtype;
    std::optional<bool> use_2d_core_grid;

    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::normalization
