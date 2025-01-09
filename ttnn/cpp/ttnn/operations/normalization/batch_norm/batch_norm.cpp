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
    if (training) {
        Tensor batch_mean = mean_NHW(input, memory_config);
        Tensor mean_sq = mean_NHW(ttnn::square(input, memory_config), memory_config);
        Tensor batch_var =
            ttnn::subtract(mean_sq, ttnn::square(batch_mean, memory_config), std::nullopt, memory_config);
        return ttnn::prim::batch_norm(
            input,
            batch_mean,
            batch_var,
            eps,
            momentum,
            training,
            weight,
            bias,
            running_mean,
            running_var,
            output,
            memory_config);
    }
    TT_FATAL(
        (running_mean.has_value() && running_var.has_value()),
        "running_mean and running_var must be defined in evaluation mode");
    return ttnn::prim::batch_norm(
        input,
        running_mean.value(),
        running_var.value(),
        eps,
        momentum,
        training,
        weight,
        bias,
        std::nullopt,
        std::nullopt,
        output,
        memory_config);
}
}  // namespace ttnn::operations::normalization
