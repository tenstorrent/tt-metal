// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "nlp_create_qkv_heads_falcon7b_device_operation_types.hpp"
#include "nlp_create_qkv_heads_falcon7b_program_factory.hpp"

namespace ttnn::operations::experimental::transformer::qkv_heads_falcon7b {

struct NlpCreateHeadsFalcon7BDeviceOperation {
    using operation_attributes_t = QkvHeadsFalcon7bParams;
    using tensor_args_t = QkvHeadsFalcon7bInputs;
    using spec_return_value_t = qkv_heads_falcon7b::spec_return_value_t;
    using tensor_return_value_t = qkv_heads_falcon7b::tensor_return_value_t;
    using program_factory_t = std::variant<NlpCreateQkvHeadsFalcon7BProgramFactory>;
    using shared_variables_t = NlpCreateQkvHeadsFalcon7BProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer::qkv_heads_falcon7b

namespace ttnn::prim {
ttnn::operations::experimental::transformer::qkv_heads_falcon7b::tensor_return_value_t nlp_create_qkv_heads_falcon7b(
    const Tensor& input, const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
}  // namespace ttnn::prim
