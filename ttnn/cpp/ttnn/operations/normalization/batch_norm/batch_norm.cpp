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
    std::optional<Tensor> weight,
    std::optional<Tensor> bias,
    std::optional<Tensor> output,
    const std::optional<MemoryConfig>& memory_config) {
    // TODO: Implementation for training mode is in progress
    TT_FATAL((!training), "Support currently provided for inference mode only");
    TT_FATAL(
        (running_mean.has_value() && running_var.has_value()),
        "running_mean and running_var must be defined in evaluation mode");
    return ttnn::prim::batch_norm(
        input, running_mean.value(), running_var.value(), eps, weight, bias, output, memory_config);
}
}  // namespace ttnn::operations::normalization
