// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm.hpp"

#include "device/batch_norm_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::normalization {

Tensor BatchNorm::invoke(
    const Tensor& input,
    std::optional<Tensor> running_mean,
    std::optional<Tensor> running_var,
    const bool training,
    const float eps,
    const float momentum,
    std::optional<Tensor> weight,
    std::optional<Tensor> bias,
    std::optional<Tensor> output,
    const std::optional<MemoryConfig>& memory_config) {
    TT_FATAL(
        (running_mean.has_value() && running_var.has_value() && (!training)),
        "running_mean and running_var must be defined in evaluation mode");
    return ttnn::prim::batch_norm(
        input, running_mean.value(), running_var.value(), eps, momentum, training, weight, bias, output, memory_config);
}
}  // namespace ttnn::operations::normalization
