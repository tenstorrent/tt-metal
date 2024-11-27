// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sgd.hpp"

#include "ttnn/operations/moreh/moreh_sgd/device/moreh_sgd_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::moreh::moreh_sgd {
std::vector<std::optional<Tensor>> MorehSgd::invoke(
    const Tensor& param_in,
    const Tensor& grad,
    const std::optional<Tensor>& momentum_buffer_in,
    const std::optional<Tensor>& param_out,
    const std::optional<Tensor>& momentum_buffer_out,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool momentum_initialized,
    const std::optional<MemoryConfig>& param_out_memory_config,
    const std::optional<MemoryConfig>& momentum_buffer_out_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::moreh_sgd(
        param_in,
        grad,
        momentum_buffer_in,
        param_out,
        momentum_buffer_out,
        lr,
        momentum,
        dampening,
        weight_decay,
        nesterov,
        momentum_initialized,
        param_out_memory_config,
        momentum_buffer_out_memory_config,
        compute_kernel_config);
}

OptionalTensors MorehSgd::create_async_optional_output_tensors(
    const Tensor& param_in,
    const Tensor& grad,
    const std::optional<Tensor>& momentum_buffer_in,
    const std::optional<Tensor>& param_out,
    const std::optional<Tensor>& momentum_buffer_out,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool momentum_initialized,
    const std::optional<MemoryConfig>& param_out_memory_config,
    const std::optional<MemoryConfig>& momentum_buffer_out_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        std::optional<Tensor>(operation::get_workers_for_op_output({param_in, grad}, {momentum_buffer_in})),
        (momentum != 0.0f)
            ? std::optional<Tensor>(operation::get_workers_for_op_output({param_in, grad}, {momentum_buffer_in}))
            : std::nullopt};
}

}  // namespace ttnn::operations::moreh::moreh_sgd
