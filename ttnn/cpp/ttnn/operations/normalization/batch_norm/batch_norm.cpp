// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm.hpp"

#include "device/batch_norm_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::normalization {
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
    return ttnn::prim::batch_norm(
        input,
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
