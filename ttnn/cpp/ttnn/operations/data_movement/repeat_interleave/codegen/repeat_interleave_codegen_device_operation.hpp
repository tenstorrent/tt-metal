// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_device_operation_types.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_program_factory.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct RepeatInterleaveCodegenDeviceOperation {
    using operation_attributes_t = RepeatInterleaveCodegenParams;
    using tensor_args_t = RepeatInterleaveCodegenInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = RepeatInterleaveCodegenProgramFactory;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

RepeatInterleaveCodegenDeviceOperation::tensor_return_value_t repeat_interleave_codegen(
    const Tensor& input, const RepeatInterleaveCodegenParams& params);

}  // namespace ttnn::prim
