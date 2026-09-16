// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <cstdint>
#include "moe_fused_swiglu_program_factory.hpp"
#include "moe_fused_swiglu_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::fused {

struct MoeFusedSwiGluDeviceOperation {
    using operation_attributes_t = OperationArguments;
    using tensor_args_t = TensorArguments;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_arguments, const tensor_args_t& tensor_arguments);
    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_arguments, const tensor_args_t& tensor_arguments);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_arguments, const tensor_args_t& tensor_arguments);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_arguments, const tensor_args_t& tensor_arguments);
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_arguments,
        const tensor_args_t& tensor_arguments,
        tensor_return_value_t& output);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::fused
