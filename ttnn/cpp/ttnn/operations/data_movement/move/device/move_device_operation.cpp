// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "move_device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement::move {

MoveDeviceOperation::program_factory_t MoveDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    switch (operation_attributes.move_op_parallelization_strategy) {
        case MoveOpParallelizationStrategy::MULTI_CORE_SHARDED: return program::MoveShardedProgramFactory{};
        case MoveOpParallelizationStrategy::MULTI_CORE_OVERLAP: return program::MoveOverlapProgramFactory{};
        case MoveOpParallelizationStrategy::MULTI_CORE: return program::MoveProgramFactory{};
        default: TT_FATAL(false, "Invalid move operation parallelization strategy");
    }
}

void MoveDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // TODO: #33357
}

void MoveDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // TODO: #33357
}

MoveDeviceOperation::spec_return_value_t MoveDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Output spec is same as output tensor spec
    return tensor_args.output_tensor.tensor_spec();
}

MoveDeviceOperation::tensor_return_value_t MoveDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Output tensor is already created and passed in tensor_args
    return tensor_args.output_tensor;
}

std::tuple<MoveDeviceOperation::operation_attributes_t, MoveDeviceOperation::tensor_args_t> MoveDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const MoveOpParallelizationStrategy& move_op_parallelization_strategy) {
    return {
        operation_attributes_t{
            .output_mem_config = output_mem_config,
            .move_op_parallelization_strategy = move_op_parallelization_strategy},
        tensor_args_t{.input_tensor = input_tensor, .output_tensor = output_tensor}};
}

}  // namespace ttnn::operations::data_movement::move
