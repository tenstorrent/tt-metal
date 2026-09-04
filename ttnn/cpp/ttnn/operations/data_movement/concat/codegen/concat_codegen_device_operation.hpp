// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include <tt_stl/reflection.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_program_factory.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct ConcatCodegenDeviceOperation {
    using operation_attributes_t = ConcatCodegenParams;
    using tensor_args_t = ConcatCodegenInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<ConcatCodegenProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    // The CB plan is derived from live L1, which the attributes do not describe. Without this
    // override a program cached against a clear frontier is reused after a large L1 allocation,
    // with CB addresses that now overlap the resident tensor.
    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output_tensor);
};

ConcatCodegenDeviceOperation::tensor_return_value_t concat_codegen(
    const std::vector<Tensor>& input_tensors, const ConcatCodegenParams& params);

}  // namespace ttnn::prim
