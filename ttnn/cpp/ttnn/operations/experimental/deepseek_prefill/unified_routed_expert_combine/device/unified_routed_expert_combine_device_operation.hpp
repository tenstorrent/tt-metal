// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "unified_routed_expert_combine_types.hpp"
#include "unified_routed_expert_combine_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine {

// The routed expert and combine_fabric2d as ONE program per chip: see UnifiedRoutedExpertCombineProgramFactory.
// Returns combine's output.
struct UnifiedRoutedExpertCombineDeviceOperation {
    using operation_attributes_t = UnifiedRoutedExpertCombineParams;
    using tensor_args_t = UnifiedRoutedExpertCombineInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<UnifiedRoutedExpertCombineProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine
