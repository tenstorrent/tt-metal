// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "roll_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::prim {

RollDeviceOperation::program_factory_t RollDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return RollShardedProgramFactory{};
}

namespace {

void validate_roll(const RollParams& operation_attributes, const RollInputs& tensor_args) {
    const Tensor& input = tensor_args.input;
    TT_FATAL(input.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operands to roll need to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(input.is_sharded(), "Native sharded roll requires a sharded input");
    TT_FATAL(operation_attributes.output_mem_config.is_sharded(), "Native sharded roll requires a sharded output");
    // The program factory enumerates a single rectangular grid row-major; a multi-range grid
    // would map cells to the wrong cores.
    TT_FATAL(
        input.shard_spec().value().grid.ranges().size() == 1,
        "Native sharded roll requires a single contiguous rectangular CoreRange");
}

}  // namespace

void RollDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_roll(operation_attributes, tensor_args);
}

void RollDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_roll(operation_attributes, tensor_args);
}

RollDeviceOperation::spec_return_value_t RollDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    // Roll preserves shape and the input's sharded layout.
    return TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(
            input.dtype(), tt::tt_metal::PageConfig(input.layout()), operation_attributes.output_mem_config));
}

RollDeviceOperation::tensor_return_value_t RollDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> RollDeviceOperation::create_op_performance_model(
    const operation_attributes_t&, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    int ideal_dev_clock_cycles = operations::data_movement::common_tm_bw_model(input_tensor, output_tensor);
    return tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t>(
        {input_tensor}, output_tensor, ideal_dev_clock_cycles);
}

RollDeviceOperation::tensor_return_value_t roll_sharded(
    const Tensor& input, uint32_t shift, int32_t dim, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = RollDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.shift = shift, .dim = dim, .output_mem_config = output_mem_config},
        OperationType::tensor_args_t{input});
}

}  // namespace ttnn::prim
