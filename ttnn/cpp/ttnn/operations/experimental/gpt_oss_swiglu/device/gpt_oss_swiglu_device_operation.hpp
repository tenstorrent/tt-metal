// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "gpt_oss_swiglu_device_operation_types.hpp"
#include "gpt_oss_swiglu_program_factory.hpp"

namespace ttnn::operations::experimental::gpt_oss_swiglu {

struct GptOssSwigluDeviceOperation {
    using operation_attributes_t = gpt_oss_swiglu::operation_attributes_t;
    using tensor_args_t = gpt_oss_swiglu::tensor_args_t;
    using tensor_return_value_t = gpt_oss_swiglu::tensor_return_value_t;
    using spec_return_value_t = gpt_oss_swiglu::spec_return_value_t;

    using program_factory_t = std::variant<program::GptOssSwigluProgramFactory>;

    static program_factory_t select_program_factory(
        [[maybe_unused]] const operation_attributes_t& attrs, [[maybe_unused]] const tensor_args_t& tensor_args) {
        return program::GptOssSwigluProgramFactory{};
    }

    static void validate_on_program_cache_miss(const operation_attributes_t& attrs, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_hit(const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& gate_tensor,
        const Tensor& up_tensor,
        float alpha,
        float clamp_limit,
        const std::optional<MemoryConfig>& output_memory_config);
};

}  // namespace ttnn::operations::experimental::gpt_oss_swiglu

namespace ttnn::experimental {

Tensor gpt_oss_swiglu(
    const ttnn::Tensor& gate_tensor,
    const ttnn::Tensor& up_tensor,
    float alpha = 1.702f,
    float clamp_limit = 7.0f,
    const std::optional<ttnn::MemoryConfig>& output_memory_config = std::nullopt);

}  // namespace ttnn::experimental
