
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attention_softmax.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"

namespace ttnn::operations::transformer {

template <bool in_place>
ttnn::Tensor ExecuteAttentionSoftmax<in_place>::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<int>& head_size_arg,
    const std::optional<const ttnn::Tensor>& attention_mask,
    const ttnn::operations::normalization::SoftmaxProgramConfig& program_config,
    const std::optional<bool> causal_mask,
    const std::optional<ttnn::MemoryConfig>& memory_config) {

    float head_size = head_size_arg.has_value() ? 1.0f / std::sqrt(head_size_arg.value()) : 1.0f;
    if constexpr (in_place) {
        TT_FATAL(attention_mask.has_value(),
         "Cannot apply divide by sqrt(head_size) using in-place version when attention_mask is not set.");
    } else {
        if (not attention_mask.has_value()) {
            auto output_tensor = ttnn::multiply(input_tensor, head_size);
            return ttnn::operations::normalization::softmax(output_tensor, memory_config.value_or(input_tensor.memory_config()));
        }
    }

    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, false, false);
    auto output_tensor = operation::run(
                            ttnn::operations::normalization::Softmax{
                                    head_size,
                                    in_place,
                                    memory_config.value_or(input_tensor.memory_config()),
                                    program_config,
                                    causal_mask.value(),
                                    kernel_config_val,
                                    false},
                                {input_tensor},
                                {attention_mask})
                                .at(0);
    return output_tensor;
}


template struct ExecuteAttentionSoftmax<false>;
template struct ExecuteAttentionSoftmax<true>;

}  // namespace ttnn::operations::transformer
