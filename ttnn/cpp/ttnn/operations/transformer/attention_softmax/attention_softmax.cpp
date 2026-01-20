
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "attention_softmax.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/normalization/softmax/device/softmax_device_operation.hpp"
#include "ttnn/operations/normalization/softmax/device/softmax_operation_types.hpp"

namespace ttnn::operations::transformer {

template <bool in_place>
ttnn::Tensor ExecuteAttentionSoftmax<in_place>::invoke(
    ttnn::Tensor& input_tensor,
    const std::optional<int>& head_size_arg,
    const std::optional<const ttnn::Tensor>& attention_mask,
    const ttnn::SoftmaxProgramConfig& /*program_config*/,
    const std::optional<bool> causal_mask,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    const float head_size = head_size_arg.has_value() ? 1.0f / std::sqrt(head_size_arg.value()) : 1.0f;
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;

    // TODO: switch to stable softmax once accuracy issue in tutorial is fixed
    // See issue: #28525
    const bool numeric_stable = false;

    if constexpr (in_place) {
        TT_FATAL(
            attention_mask.has_value(),
            "Cannot apply divide by sqrt(head_size) using in-place version when attention_mask is not set.");
        return prim::scale_mask_softmax_in_place(
            input_tensor,
            head_size,
            attention_mask,
            ttnn::SoftmaxDefaultProgramConfig{},
            causal_mask.value_or(false),
            compute_kernel_config,
            numeric_stable);
    } else {
        if (not attention_mask.has_value()) {
            auto output_tensor = ttnn::multiply(input_tensor, head_size);
            return prim::softmax(output_tensor, -1, memory_config.value_or(input_tensor.memory_config()));
        }
    }

    return prim::scale_mask_softmax(
        input_tensor,
        head_size,
        attention_mask,
        memory_config.value_or(input_tensor.memory_config()),
        causal_mask.value_or(false),
        compute_kernel_config,
        numeric_stable);
}

template struct ExecuteAttentionSoftmax<false>;
template struct ExecuteAttentionSoftmax<true>;

}  // namespace ttnn::operations::transformer
