// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/codegen/reshape_codegen_program_factory.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct ReshapeCodegenDeviceOperation {
    using operation_attributes_t = ReshapeCodegenParams;
    using tensor_args_t = ReshapeCodegenInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<ReshapeCodegenRmProgramFactory, ReshapeCodegenTileProgramFactory>;

    static program_factory_t select_program_factory(
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

ReshapeCodegenDeviceOperation::tensor_return_value_t reshape_codegen(
    const Tensor& input,
    const ttnn::Shape& output_logical_shape,
    const ttnn::Shape& output_padded_shape,
    const ReshapeCodegenParams& params,
    std::optional<Tensor> optional_output_tensor = std::nullopt);

}  // namespace ttnn::prim
