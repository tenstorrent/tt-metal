// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/data_movement/repeat/device/repeat_device_operation_types.hpp"
#include "ttnn/operations/data_movement/repeat/device/repeat_program_factory_last_dim.hpp"
#include "ttnn/operations/data_movement/repeat/device/repeat_program_factory_higher_dim.hpp"

namespace ttnn::operations::data_movement::repeat {

struct RepeatDeviceOperation {
    using operation_attributes_t = RepeatParams;
    using tensor_args_t = RepeatInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t =
        std::variant<program::RepeatProgramFactoryLastDim, program::RepeatProgramFactoryHigherDim>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& input_tensors);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& input_tensors);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output_tensor);
};
}  // namespace ttnn::operations::data_movement::repeat

namespace ttnn::prim {
ttnn::operations::data_movement::repeat::RepeatDeviceOperation::tensor_return_value_t repeat(
    const Tensor& input,
    uint32_t m_num_repeats,
    bool m_is_last_dim,
    const tt::tt_metal::MemoryConfig& output_mem_config);
}  // namespace ttnn::prim
