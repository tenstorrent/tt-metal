// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "move_device_operation_types.hpp"
#include "move_program_factory.hpp"
#include "move_overlap_program_factory.hpp"
#include "move_sharded_program_factory.hpp"

namespace ttnn::operations::data_movement::move {

struct MoveDeviceOperation {
    // Type aliases
    using operation_attributes_t = move::operation_attributes_t;
    using tensor_args_t = move::tensor_args_t;
    using tensor_return_value_t = Tensor;
    using spec_return_value_t = ttnn::TensorSpec;

    using program_factory_t = std::
        variant<program::MoveProgramFactory, program::MoveOverlapProgramFactory, program::MoveShardedProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::data_movement::move

namespace ttnn::prim {
ttnn::operations::data_movement::move::MoveDeviceOperation::tensor_return_value_t move(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const ttnn::operations::data_movement::move::MoveOpParallelizationStrategy& move_op_parallelization_strategy);
}  // namespace ttnn::prim
