// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax.hpp"

#include "device/softmax_operation_types.hpp"
#include "device/softmax_device_operation.hpp"

#include <algorithm>

#include <tt_stl/assert.hpp>
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

constexpr float DEFAULT_SCALE_VALUE = 1.0f;

namespace ttnn {

namespace {

// Rank <= 4 is padded with leading 1s. Rank > 4 keeps dim 0 and the last two dims and folds the dims in between
// into one, which is exact for softmax over the last dim (the batch, head and tile structure is unchanged).
Tensor fold_to_4D(const Tensor& tensor) {
    const auto& logical = tensor.logical_shape();
    const auto rank = logical.rank();
    if (rank <= 4) {
        return ttnn::unsqueeze_to_4D(tensor);
    }
    const auto& padded = tensor.padded_shape();
    uint32_t mid_logical = 1;
    uint32_t mid_padded = 1;
    for (size_t i = 1; i + 2 < rank; ++i) {
        mid_logical *= logical[i];
        mid_padded *= padded[i];
    }
    return ttnn::reshape(
        tensor,
        ttnn::Shape({logical[0], mid_logical, logical[rank - 2], logical[rank - 1]}),
        ttnn::Shape({padded[0], mid_padded, padded[rank - 2], padded[rank - 1]}));
}

// A mask keeps its own batch dim 0; its dims between batch and height must be 1 and fold to a single 1.
std::optional<const Tensor> fold_mask_to_4D(const std::optional<const Tensor>& mask) {
    if (mask.has_value() && mask->logical_shape().rank() > 4) {
        return fold_to_4D(*mask);
    }
    return mask;
}

}  // namespace

Tensor softmax(
    const Tensor& input_tensor,
    int dim,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto mem_config = memory_config.value_or(input_tensor.memory_config());
    const auto& input_shape = input_tensor.logical_shape();
    const int rank = static_cast<int>(input_shape.size());
    // A scalar has the single virtual axis 0 / -1, as in torch.
    const int dim_rank = std::max(rank, 1);
    const int dim_calculated = dim < 0 ? dim_rank + dim : dim;
    if (dim_calculated < 0 || dim_calculated >= dim_rank) {
        TT_THROW(
            "Dimension out of range (expected to be in range of [{}, {}], but got {})", -dim_rank, dim_rank - 1, dim);
    }

    // Early exit for empty tensors
    if ((rank == 0) || (input_tensor.logical_volume() == 0)) {
        return ttnn::full(
            input_shape,
            DEFAULT_SCALE_VALUE,
            input_tensor.dtype(),
            input_tensor.layout(),
            *input_tensor.device(),
            mem_config);
    }

    // Operation
    auto output_tensor = prim::softmax(
        input_tensor, static_cast<int8_t>(dim_calculated), mem_config, compute_kernel_config, numeric_stable);

    return output_tensor.reshape(input_shape);
}

Tensor scale_mask_softmax(
    const Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    const std::optional<MemoryConfig>& memory_config,
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
    auto input_tensor_4D = fold_to_4D(input_tensor);
    auto output_tensor = prim::scale_mask_softmax(
        input_tensor_4D,
        scale,
        fold_mask_to_4D(mask),
        memory_config.value_or(input_tensor.memory_config()),
        is_causal_mask,
        compute_kernel_config,
        numeric_stable);

    return output_tensor.reshape(input_shape);
}

Tensor softmax_in_place(
    const Tensor& input_tensor,
    int dim,
    const SoftmaxProgramConfig& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto& input_shape = input_tensor.logical_shape();
    const int rank = static_cast<int>(input_shape.size());
    // A scalar has the single virtual axis 0 / -1, as in torch.
    const int dim_rank = std::max(rank, 1);
    const int dim_calculated = dim < 0 ? dim_rank + dim : dim;
    if (dim_calculated < 0 || dim_calculated >= dim_rank) {
        TT_THROW(
            "Dimension out of range (expected to be in range of [{}, {}], but got {})", -dim_rank, dim_rank - 1, dim);
    }

    // Early exit for empty tensors
    if ((rank == 0) || (input_tensor.logical_volume() == 0)) {
        // Fill the tensor with the default scale value
        return ttnn::fill(input_tensor, DEFAULT_SCALE_VALUE);
    }

    // Operation
    auto input_tensor_4D = fold_to_4D(input_tensor);
    // The reshape adds leading dims (rank < 4) or folds the middle dims (rank > 4), so the softmax dim moves too.
    // Only the last two dims survive a fold.
    if (rank > 4 && dim_calculated < rank - 2) {
        TT_THROW("softmax_in_place on a rank {} tensor supports only the last two dims, got dim {}", rank, dim);
    }
    const int dim_4D = rank > 4 ? dim_calculated - (rank - 4) : dim_calculated + 4 - rank;
    auto output_tensor = prim::softmax_in_place(
        input_tensor_4D, static_cast<int8_t>(dim_4D), program_config, compute_kernel_config, numeric_stable);

    return output_tensor.reshape(input_shape);
}

Tensor scale_mask_softmax_in_place(
    const Tensor& input_tensor,
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
    auto input_tensor_4D = fold_to_4D(input_tensor);
    auto output_tensor = prim::scale_mask_softmax_in_place(
        input_tensor_4D,
        scale,
        fold_mask_to_4D(mask),
        program_config,
        is_causal_mask,
        compute_kernel_config,
        numeric_stable);

    return output_tensor.reshape(input_shape);
}

Tensor scale_causal_mask_hw_dims_softmax_in_place(
    const Tensor& input_tensor,
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
    auto output_tensor = prim::scale_causal_mask_hw_dims_softmax_in_place(
        input_tensor_4D, scale, mask, program_config, compute_kernel_config, numeric_stable);

    return output_tensor.reshape(input_shape);
}

}  // namespace ttnn
