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

namespace ttnn::experimental::prim {

struct NlpCreateHeadsFalcon7BDeviceOperation {
    using operation_attributes_t = NlpCreateQkvHeadsFalcon7bParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = NlpCreateQkvHeadsFalcon7bResultSpec;
    using tensor_return_value_t = NlpCreateQkvHeadsFalcon7bResult;
    using program_factory_t = std::variant<NlpCreateQkvHeadsFalcon7BProgramFactory>;
    using shared_variables_t = NlpCreateQkvHeadsFalcon7BProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
ttnn::experimental::prim::NlpCreateQkvHeadsFalcon7bResult nlp_create_qkv_heads_falcon7b(
    const Tensor& input, const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
}  // namespace ttnn::prim
