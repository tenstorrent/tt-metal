// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax.hpp"

#include "cpp/ttnn/operations/moreh/moreh_softmax/device/moreh_softmax_device_operation.hpp"
#include "device/softmax_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"

namespace ttnn::operations::normalization {

using namespace moreh::moreh_softmax;

ttnn::Tensor ExecuteSoftmax::invoke(
    const ttnn::Tensor& input_tensor,
    const int dim_arg,
    const std::optional<ttnn::MemoryConfig>& memory_config_arg,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const bool numeric_stable) {
    auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
    const auto& input_shape = input_tensor.logical_shape();
    auto rank = input_shape.size();
    auto dim = dim_arg;
    if (dim < 0) {
        dim = rank + dim;
    }

    // For 0D or 0V tensors
    if ((rank == 0) || (input_tensor.logical_volume() == 0)) {
        return ttnn::full(
            input_shape, 1.0f, input_tensor.dtype(), input_tensor.layout(), *input_tensor.mesh_device(), memory_config);
    }

    if (rank > 4) {
        auto output_tensor = ttnn::prim::moreh_softmax(
            input_tensor,
            dim,
            std::nullopt,
            MorehSoftmaxOp::SOFTMAX,
            MorehSoftmaxOpParallelizationStrategy::NONE,
            memory_config,
            compute_kernel_config);
        return ttnn::reshape(output_tensor, input_shape);
    }

    auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    if (dim == rank - 1) {
        auto output_tensor = ttnn::operations::normalization::softmax(
            input_tensor_4D, memory_config, compute_kernel_config, numeric_stable);
        return ttnn::reshape(output_tensor, input_shape);
    } else {
        auto dim_4D = dim + 4 - rank;
        auto output_tensor = ttnn::prim::moreh_softmax(
            input_tensor_4D,
            dim_4D,
            std::nullopt,
            MorehSoftmaxOp::SOFTMAX,
            MorehSoftmaxOpParallelizationStrategy::NONE,
            memory_config,
            compute_kernel_config);
        return ttnn::reshape(output_tensor, input_shape);
    }
}

ttnn::Tensor ExecuteScaleMaskSoftmax::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const bool is_causal_mask,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const bool numeric_stable) {
    const auto& input_shape = input_tensor.logical_shape();
    const auto rank = input_shape.size();

    // For 0D or 0V tensors
    if ((rank == 0) || (input_tensor.logical_volume() == 0)) {
        return ttnn::full(
            input_shape,
            scale.value_or(1.0f),
            input_tensor.dtype(),
            input_tensor.layout(),
            *input_tensor.mesh_device(),
            memory_config);
    }

    auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    auto output_tensor = ttnn::operations::normalization::scale_mask_softmax(
        input_tensor_4D,
        scale,
        mask,
        memory_config.value_or(input_tensor.memory_config()),
        is_causal_mask,
        compute_kernel_config,
        numeric_stable);
    return ttnn::reshape(output_tensor, input_shape);
}

ttnn::Tensor ExecuteSoftmaxInPlace::invoke(
    const ttnn::Tensor& input_tensor,
    const SoftmaxProgramConfig& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const bool numeric_stable) {
    const auto& input_shape = input_tensor.logical_shape();
    const auto rank = input_shape.size();

    // For 0D or 0V tensors
    if ((rank == 0) || (input_tensor.logical_volume() == 0)) {
        // Fill the tensor with 1.0f
        return ttnn::fill(input_tensor, 1.0f);
    }

    auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    auto output_tensor = ttnn::operations::normalization::softmax_in_place(
        input_tensor_4D, program_config, compute_kernel_config, numeric_stable);
    return ttnn::reshape(output_tensor, input_shape);
}

ttnn::Tensor ExecuteScaleMaskSoftmaxInPlace::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    const SoftmaxProgramConfig& program_config,
    const bool is_causal_mask,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const bool numeric_stable) {
    const auto& input_shape = input_tensor.logical_shape();
    auto rank = input_shape.size();

    // For 0D or 0V tensors
    if ((rank == 0) || (input_tensor.logical_volume() == 0)) {
        // Fill the tensor with 1.0f
        return ttnn::fill(input_tensor, scale.value_or(1.0f));
    }

    auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    auto output_tensor = ttnn::operations::normalization::scale_mask_softmax_in_place(
        input_tensor_4D, scale, mask, program_config, is_causal_mask, compute_kernel_config, numeric_stable);
    return ttnn::reshape(output_tensor, input_shape);
}

ttnn::Tensor ExecuteScaleCausalMaskHWSoftmaxInPlace::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    const SoftmaxProgramConfig& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const bool numeric_stable) {
    const auto& input_shape = input_tensor.logical_shape();
    auto rank = input_shape.size();

    // For 0D or 0V tensors
    if ((rank == 0) || (input_tensor.logical_volume() == 0)) {
        // Fill the tensor with 1.0f
        return ttnn::fill(input_tensor, scale.value_or(1.0f));
    }

    auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    auto output_tensor = ttnn::operations::normalization::scale_causal_mask_hw_dims_softmax_in_place(
        input_tensor_4D, scale, mask, program_config, compute_kernel_config, numeric_stable);
    return ttnn::reshape(output_tensor, input_shape);
}

}  // namespace ttnn::operations::normalization
