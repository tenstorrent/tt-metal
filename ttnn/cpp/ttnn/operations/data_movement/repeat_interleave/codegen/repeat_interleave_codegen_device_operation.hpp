// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_device_operation_types.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_program_factory.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct RepeatInterleaveCodegenDeviceOperation {
    using operation_attributes_t = RepeatInterleaveCodegenParams;
    using tensor_args_t = RepeatInterleaveCodegenInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<RepeatInterleaveCodegenProgramFactory>;

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
    const Tensor& input,
    uint32_t rep_dim,
    uint32_t num_repeats,
    uint32_t lower_pages,
    uint32_t rep_dim_pages,
    uint32_t total_out_pages,
    uint32_t stick_size,
    uint32_t stick_size_out,
    const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::prim
