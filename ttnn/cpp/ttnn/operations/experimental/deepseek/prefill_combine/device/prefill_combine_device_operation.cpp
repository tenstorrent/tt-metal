// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "prefill_combine_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::prefill_combine {

void PrefillCombineDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // TODO: Implement validation
    // - Check tensor layouts (ROW_MAJOR expected)
    // - Check tensor dtypes
    // - Check tensor shapes are compatible
    // - Validate configuration parameters
}

void PrefillCombineDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // Empty for now
}

PrefillCombineDeviceOperation::spec_return_value_t PrefillCombineDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // TODO: Compute output specs based on input shapes and operation attributes
    // Return spec for combined output tensor

    // Stub: throw not implemented
    TT_THROW("compute_output_specs not yet implemented");
}

PrefillCombineDeviceOperation::tensor_return_value_t PrefillCombineDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // TODO: Create output tensor based on computed specs

    // Stub: throw not implemented
    TT_THROW("create_output_tensors not yet implemented");
}

}  // namespace ttnn::operations::experimental::deepseek::prefill_combine
