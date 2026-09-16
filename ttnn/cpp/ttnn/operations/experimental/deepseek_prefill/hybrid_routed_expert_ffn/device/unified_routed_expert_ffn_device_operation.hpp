// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <variant>

#include "unified_routed_expert_ffn_program_factory.hpp"
#include "unified_routed_expert_ffn_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::unified {

struct UnifiedRoutedExpertFfnDeviceOperation {
    using operation_attributes_t = UnifiedRoutedExpertFfnParams;
    using tensor_args_t = UnifiedRoutedExpertFfnInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    // Descriptor-based, matching the other carried implementation: the merged op composes
    // both halves into one ProgramDescriptor, which a program_factory_t variant cannot express.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::unified
