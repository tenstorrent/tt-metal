// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ShardedToInterleavedDeviceOperation::program_factory_t ShardedToInterleavedDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::ShardedToInterleavedProgramFactory{};
}

void ShardedToInterleavedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto shard_spec = input_tensor.shard_spec().value();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.memory_config().is_sharded(), "Input tensor must be sharded");
    TT_FATAL(input_tensor.memory_config().buffer_type() == BufferType::L1, "Input tensor must be in L1");

    if (tensor_args.preallocated_output.has_value()) {
        const auto& output_tensor = tensor_args.preallocated_output.value();
        TT_FATAL(output_tensor.memory_config() == args.output_mem_config, "Mismatched output memory config");
        TT_FATAL(output_tensor.dtype() == args.output_dtype, "Mismatched output dtype");
        TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
        TT_FATAL(output_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
        TT_FATAL(output_tensor.device() == input_tensor.device(), "Operands to shard need to be on the same device!");
    }

    TT_FATAL(
        args.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Output memory config must be Interleaved");

    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        uint32_t l1_alignment = hal::get_l1_alignment();
        TT_FATAL(
            (*input_tensor.memory_config().shard_spec()).shape[1] * input_tensor.element_size() % (l1_alignment) == 0,
            "Shard page size must be aligned to {}B for L1 Tensor",
            l1_alignment);
    }

    if (input_tensor.dtype() != args.output_dtype) {
        TT_FATAL(input_tensor.layout() == Layout::TILE, "If diff output type, tensor must be TILED");
    }
}

void ShardedToInterleavedDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

ShardedToInterleavedDeviceOperation::spec_return_value_t ShardedToInterleavedDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    return TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            args.output_dtype,
            PageConfig(input_tensor.layout()),
            args.output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()));
}

ShardedToInterleavedDeviceOperation::tensor_return_value_t ShardedToInterleavedDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    auto spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(spec, input_tensor.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<ShardedToInterleavedDeviceOperation::tensor_return_value_t>
ShardedToInterleavedDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) const {
    int ideal_dev_clock_cycles = common_tm_bw_model(tensor_args.input_tensor, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {tensor_args.input_tensor}, {output_tensor}, ideal_dev_clock_cycles);
    return result;
}
}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::ShardedToInterleavedDeviceOperation::tensor_return_value_t sharded_to_interleaved(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    const std::optional<Tensor>& preallocated_output) {
    using OperationType = ttnn::operations::data_movement::ShardedToInterleavedDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_mem_config = output_mem_config, .output_dtype = output_dtype, .num_slices = 1, .slice_index = 0},
        OperationType::tensor_args_t{.input_tensor = input_tensor, .preallocated_output = preallocated_output});
}
}  // namespace ttnn::prim
