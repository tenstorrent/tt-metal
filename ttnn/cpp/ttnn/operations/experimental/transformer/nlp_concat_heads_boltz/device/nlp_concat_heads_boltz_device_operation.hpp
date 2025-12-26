// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "nlp_concat_heads_boltz_device_operation_types.hpp"
#include "nlp_concat_heads_boltz_program_factory.hpp"

namespace ttnn::operations::experimental::nlp_concat_heads_boltz {

struct NLPConcatHeadsBoltzDeviceOperation {
    using operation_attributes_t = nlp_concat_heads_boltz::operation_attributes_t;
    using tensor_args_t = nlp_concat_heads_boltz::tensor_args_t;
    using spec_return_value_t = nlp_concat_heads_boltz::spec_return_value_t;
    using tensor_return_value_t = nlp_concat_heads_boltz::tensor_return_value_t;
    using program_factory_t = std::variant<NLPConcatHeadsBoltzProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::nlp_concat_heads_boltz

namespace ttnn::prim {
ttnn::operations::experimental::nlp_concat_heads_boltz::tensor_return_value_t nlp_concat_heads_boltz(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& memory_config,
    std::optional<Tensor> optional_output_tensor);
}  // namespace ttnn::prim
