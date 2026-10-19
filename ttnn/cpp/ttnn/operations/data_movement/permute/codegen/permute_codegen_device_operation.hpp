// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt_stl/reflection.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/permute/codegen/permute_codegen_program_factory.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct PermuteCodegenDeviceOperation {
    using operation_attributes_t = PermuteCodegenParams;
    using tensor_args_t = PermuteCodegenInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<PermuteCodegenProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output_tensor);
};

PermuteCodegenDeviceOperation::tensor_return_value_t permute_codegen(
    const Tensor& input,
    const PermuteCodegenParams& params,
    std::optional<Tensor> optional_output_tensor = std::nullopt);

}  // namespace ttnn::prim
