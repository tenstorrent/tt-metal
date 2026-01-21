// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <tuple>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "nlp_concat_heads_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "nlp_concat_heads_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct NLPConcatHeadsDeviceOperation {
    using operation_attributes_t = NlpConcatHeadsParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<NLPConcatHeadsProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor nlp_concat_heads(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config);
}  // namespace ttnn::prim
