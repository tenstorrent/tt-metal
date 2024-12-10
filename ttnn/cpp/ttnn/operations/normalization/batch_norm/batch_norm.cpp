// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm.hpp"

#include "device/batch_norm_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_mean/device/moreh_mean_device_operation.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/unary/unary_composite.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::normalization {

inline Tensor mean_NHW(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());
    auto batch_mean = input_tensor;
    ttnn::SmallVector<int64_t> dims = {0, 2, 3};
    std::sort(dims.begin(), dims.end());
    for (uint32_t i = dims.size() - 1; i > 0; i--) {
        auto temp_output = ttnn::prim::moreh_mean(
            batch_mean, dims[i], true, std::nullopt, std::nullopt, output_memory_config, std::nullopt);
        batch_mean = temp_output;
    }
    return ttnn::prim::moreh_mean(
        batch_mean, dims.front(), true, std::nullopt, std::nullopt, output_memory_config, std::nullopt);
}

std::vector<std::optional<Tensor>> BatchNorm::invoke(
    const Tensor& input,
    const float eps,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const std::vector<bool>& are_required_outputs,
    const std::optional<const Tensor>& output,
    const std::optional<const Tensor>& mean,
    const std::optional<const Tensor>& rstd,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<MemoryConfig>& mean_memory_config,
    const std::optional<MemoryConfig>& rstd_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    // moreh mean code

    Tensor batch_mean = mean_NHW(input, memory_config);
    Tensor mean_sq = mean_NHW(ttnn::square(input, memory_config), memory_config);
    Tensor batch_var = ttnn::subtract(mean_sq, ttnn::square(batch_mean, memory_config), std::nullopt, memory_config);

    // send mean as one input and check
    return ttnn::prim::batch_norm(
        input,
        batch_mean,
        batch_var,
        eps,
        gamma,
        beta,
        are_required_outputs,
        output,
        mean,
        rstd,
        memory_config,
        mean_memory_config,
        rstd_memory_config,
        compute_kernel_config);
}

OptionalTensors BatchNorm::create_async_optional_output_tensors(
    const Tensor& input,
    const float eps,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const std::vector<bool>& are_required_outputs,
    const std::optional<const Tensor>& output,
    const std::optional<const Tensor>& mean,
    const std::optional<const Tensor>& rstd,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<MemoryConfig>& mean_memory_config,
    const std::optional<MemoryConfig>& rstd_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        are_required_outputs.at(0) ? std::optional<Tensor>(operation::get_workers_for_op_output({input}, {gamma, beta}))
                                   : std::nullopt,
        are_required_outputs.at(1) ? std::optional<Tensor>(operation::get_workers_for_op_output({input}, {gamma, beta}))
                                   : std::nullopt,
        are_required_outputs.at(2) ? std::optional<Tensor>(operation::get_workers_for_op_output({input}, {gamma, beta}))
                                   : std::nullopt};
}
}  // namespace ttnn::operations::normalization
