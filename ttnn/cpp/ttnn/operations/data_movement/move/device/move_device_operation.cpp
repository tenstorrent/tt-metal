// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "move_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::prim {

MoveDeviceOperation::program_factory_t MoveDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    switch (operation_attributes.move_op_parallelization_strategy) {
        case MoveOpParallelizationStrategy::MULTI_CORE_SHARDED: return MoveShardedProgramFactory{};
        case MoveOpParallelizationStrategy::MULTI_CORE_OVERLAP: return MoveOverlapProgramFactory{};
        case MoveOpParallelizationStrategy::MULTI_CORE: return MoveProgramFactory{};
        default: TT_FATAL(false, "Invalid move operation parallelization strategy");
    }
}

void MoveDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // TODO: #33357
}

void MoveDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // TODO: #33357
}

MoveDeviceOperation::spec_return_value_t MoveDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Output spec is same as output tensor spec
    return tensor_args.output_tensor.tensor_spec();
}

MoveDeviceOperation::tensor_return_value_t MoveDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Output tensor is already created and passed in tensor_args
    return tensor_args.output_tensor;
}

tt::tt_metal::operation::OpPerformanceModelGeneral<MoveDeviceOperation::tensor_return_value_t>
MoveDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& output_tensor = tensor_return_value;
    const int ideal_dev_clock_cycles = operations::data_movement::common_tm_bw_model(input_tensor, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, output_tensor, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::prim

namespace ttnn::prim {
ttnn::prim::MoveDeviceOperation::tensor_return_value_t move(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const ttnn::prim::MoveOpParallelizationStrategy& move_op_parallelization_strategy) {
    using OperationType = ttnn::prim::MoveDeviceOperation;
    bool backwards = false;
    if (move_op_parallelization_strategy == ttnn::prim::MoveOpParallelizationStrategy::MULTI_CORE) {
        Buffer* src_buffer = input_tensor.buffer();
        Buffer* dst_buffer = output_tensor.buffer();
        const bool src_and_dst_in_l1 = src_buffer->buffer_type() == tt::tt_metal::BufferType::L1 &&
                                       dst_buffer->buffer_type() == tt::tt_metal::BufferType::L1;
        const uint32_t src_base = src_buffer->address();
        const uint32_t dst_base = dst_buffer->address();
        const uint32_t copy_size_bytes = dst_buffer->size();
        const bool ranges_overlap = (src_base < dst_base + copy_size_bytes) && (dst_base < src_base + copy_size_bytes);
        backwards = src_and_dst_in_l1 && ranges_overlap && (dst_base > src_base);
    }
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_mem_config = output_mem_config,
            .move_op_parallelization_strategy = move_op_parallelization_strategy,
            .backwards = backwards},
        OperationType::tensor_args_t{.input_tensor = input_tensor, .output_tensor = output_tensor});
}
}  // namespace ttnn::prim
