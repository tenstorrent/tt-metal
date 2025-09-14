// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax.hpp"

#include "device/softmax_operation_types.hpp"
#include "device/softmax_device_operation.hpp"

#include "tt-metalium/assert.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"

constexpr float DEFAULT_SCALE_VALUE = 1.0f;
namespace ttnn::operations::normalization {
Tensor ExecuteSoftmax::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    int dim,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto mem_config = memory_config.value_or(input_tensor.memory_config());
    const auto& input_shape = input_tensor.logical_shape();
    const auto rank = input_shape.size();
    const auto dim_calculated = dim < 0 ? rank + dim : dim;
    if (dim_calculated < 0 || dim_calculated >= rank) {
        TT_THROW("Dimension out of range. Dim: {}", dim_calculated);
    }

    // Early exit for empty tensors
    if ((rank == 0) || (input_tensor.logical_volume() == 0)) {
        return ttnn::full(
            input_shape,
            DEFAULT_SCALE_VALUE,
            input_tensor.dtype(),
            input_tensor.layout(),
            *input_tensor.device(),
            memory_config);
    }

    // Operation
    auto output_tensor = ttnn::operations::normalization::softmax::softmax(
        queue_id, input_tensor, dim, mem_config, compute_kernel_config, numeric_stable);

    return ttnn::reshape(output_tensor, input_shape);
}

Tensor ExecuteScaleMaskSoftmax::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    bool is_causal_mask,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto& input_shape = input_tensor.logical_shape();
    const auto rank = input_shape.size();

    // Early exit for empty tensors
    if ((rank == 0) || (input_tensor.logical_volume() == 0)) {
        return ttnn::full(
            input_shape,
            scale.value_or(DEFAULT_SCALE_VALUE),
            input_tensor.dtype(),
            input_tensor.layout(),
            *input_tensor.device(),
            memory_config);
    }

    // Operation
    auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    auto output_tensor = ttnn::operations::normalization::softmax::scale_mask_softmax(
        queue_id,
        input_tensor_4D,
        scale,
        mask,
        memory_config.value_or(input_tensor.memory_config()),
        is_causal_mask,
        compute_kernel_config,
        numeric_stable);

    return ttnn::reshape(output_tensor, input_shape);
}

Tensor ExecuteSoftmaxInPlace::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    int dim,
    const SoftmaxProgramConfig& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto& input_shape = input_tensor.logical_shape();
    const auto rank = input_shape.size();
    const auto dim_calculated = dim < 0 ? rank + dim : dim;
    if (dim_calculated < 0 || dim_calculated >= rank) {
        TT_THROW("Dimension out of range. Dim: {}", dim_calculated);
    }

    // Early exit for empty tensors
    if ((rank == 0) || (input_tensor.logical_volume() == 0)) {
        // Fill the tensor with the default scale value
        return ttnn::fill(input_tensor, DEFAULT_SCALE_VALUE);
    }

    // Operation
    auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    auto output_tensor = ttnn::operations::normalization::softmax::softmax_in_place(
        queue_id, input_tensor_4D, dim_calculated, program_config, compute_kernel_config, numeric_stable);

    return ttnn::reshape(output_tensor, input_shape);
}

Tensor ExecuteScaleMaskSoftmaxInPlace::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    const SoftmaxProgramConfig& program_config,
    bool is_causal_mask,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto& input_shape = input_tensor.logical_shape();
    const auto rank = input_shape.size();

    // Early exit for empty tensors
    if ((rank == 0) || (input_tensor.logical_volume() == 0)) {
        // Fill the tensor with the default scale value
        return ttnn::fill(input_tensor, scale.value_or(DEFAULT_SCALE_VALUE));
    }

    // Operation
    auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    auto output_tensor = ttnn::operations::normalization::softmax::scale_mask_softmax_in_place(
        queue_id, input_tensor_4D, scale, mask, program_config, is_causal_mask, compute_kernel_config, numeric_stable);

    return ttnn::reshape(output_tensor, input_shape);
}

Tensor ExecuteScaleCausalMaskHWSoftmaxInPlace::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    const SoftmaxProgramConfig& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto& input_shape = input_tensor.logical_shape();
    const auto rank = input_shape.size();

    // Early exit for empty tensors
    if ((rank == 0) || (input_tensor.logical_volume() == 0)) {
        // Fill the tensor with 1.0f
        return ttnn::fill(input_tensor, scale.value_or(DEFAULT_SCALE_VALUE));
    }

    // Operation
    auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    auto output_tensor = ttnn::operations::normalization::softmax::scale_causal_mask_hw_dims_softmax_in_place(
        queue_id, input_tensor_4D, scale, mask, program_config, compute_kernel_config, numeric_stable);

    return ttnn::reshape(output_tensor, input_shape);
}
}  // namespace ttnn::operations::normalization
