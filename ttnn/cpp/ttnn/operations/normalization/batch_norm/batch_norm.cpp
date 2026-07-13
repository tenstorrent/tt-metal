// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm.hpp"

#include "device/batch_norm_device_operation.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_composite_op.hpp"
#include "device/running_statistics_device_operation.hpp"
#include "device/batch_norm_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::normalization {

inline Tensor mean_NHW(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    auto output_mem_config = memory_config.value_or(input_tensor.memory_config());
    ttsl::SmallVector<int> dims = {2, 3};
    Tensor mean_hw = ttnn::mean(input_tensor, dims, true, output_mem_config, compute_kernel_config);
    return ttnn::mean(mean_hw, 0, true, output_mem_config, compute_kernel_config);
}

}  // namespace ttnn::operations::normalization

namespace ttnn {

Tensor batch_norm(
    const Tensor& input,
    std::optional<Tensor> running_mean,
    std::optional<Tensor> running_var,
    const bool training,
    const float eps,
    const float momentum,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(
        input.logical_shape().rank() >= 4,
        "batch_norm not supported for tensors with rank < 4. (rank={})",
        input.logical_shape().rank(),
        input.logical_shape().rank());

    // output must have the same dtype as input
    if (output.has_value()) {
        TT_FATAL(
            output->dtype() == input.dtype(),
            "batch_norm: output dtype ({}) must match input dtype ({})",
            output->dtype(),
            input.dtype());
    }

    // All user-provided parameters (running_mean, running_var, weight, bias) must share the same dtype
    auto get_param_dtype = [&]() -> std::optional<DataType> {
        if (running_mean.has_value()) {
            return running_mean->dtype();
        }
        if (running_var.has_value()) {
            return running_var->dtype();
        }
        if (weight.has_value()) {
            return weight->dtype();
        }
        if (bias.has_value()) {
            return bias->dtype();
        }
        return std::nullopt;
    };
    auto param_dtype = get_param_dtype();
    if (param_dtype.has_value()) {
        auto check_param = [&](const std::optional<Tensor>& t, std::string_view name) {
            if (t.has_value()) {
                TT_FATAL(
                    t->dtype() == param_dtype.value(),
                    "batch_norm: {} dtype ({}) must match other parameter tensors dtype ({})",
                    name,
                    t->dtype(),
                    param_dtype.value());
            }
        };
        check_param(running_mean, "running_mean");
        check_param(running_var, "running_var");
        check_param(weight, "weight");
        check_param(bias, "bias");
    }

    // For 0V tensors
    if (input.logical_volume() == 0) [[unlikely]] {
        return ttnn::clone(
            input,
            /*dtype=*/std::nullopt,
            memory_config.value_or(input.memory_config()),
            /*compute_kernel_config*/ std::nullopt);
    }

    Tensor batch_mean, batch_var;
    if (training) {
        // Note: These generic TTNN ops use the compute_kernel_config as-is. In mixed precision,
        // the highest-precision accumulation is only enforced inside the batch_norm and
        // running_statistics prims (via any_float32). If required in the future, we may need to
        // propagate the precision requirement here so that the output `batch_mean`/`batch_var` are in
        // higher precision rather than inheriting the `dtype` of input.
        batch_mean = operations::normalization::mean_NHW(input, memory_config, compute_kernel_config);
        // Use the centered two-pass form E[(x - mean)^2] instead of E[x^2] - E[x]^2.
        // The latter suffers catastrophic cancellation when the batch mean is large
        // relative to the batch variance, which can drive the variance slightly
        // negative and produce NaNs in the normalization step.
        auto centered = ttnn::subtract(input, batch_mean, std::nullopt, memory_config);
        batch_var = operations::normalization::mean_NHW(
            ttnn::square(centered, memory_config), memory_config, compute_kernel_config);
    } else {
        TT_FATAL(
            (running_mean.has_value() && running_var.has_value()),
            "running_mean and running_var must be defined in evaluation mode");
        batch_mean = running_mean.value();
        batch_var = running_var.value();
    }

    // Normalize before updating running stats: running_statistics writes running_mean/running_var
    // in place, which would corrupt weight/bias if the caller aliases those buffers.
    auto output_tensor = ttnn::prim::batch_norm(
        input, batch_mean, batch_var, eps, weight, bias, output, memory_config, compute_kernel_config);

    if (training) {
        ttnn::prim::running_statistics(
            batch_mean, batch_var, momentum, running_mean, running_var, memory_config, compute_kernel_config);
    }

    return output_tensor;
}

}  // namespace ttnn
