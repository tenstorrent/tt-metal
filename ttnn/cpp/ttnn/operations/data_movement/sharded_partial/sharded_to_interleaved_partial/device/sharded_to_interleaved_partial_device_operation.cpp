// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

ShardedToInterleavedPartialDeviceOperation::program_factory_t
ShardedToInterleavedPartialDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ShardedToInterleavedPartialProgramFactory{};
}

void ShardedToInterleavedPartialDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& cache_tensor = tensor_args.cache_tensor;
    auto shard_spec = input_tensor.shard_spec().value();

    // Validate output tensor
    TT_FATAL(
        args.slice_index >= 0 && args.slice_index < args.num_slices,
        "Slice index and num_slices don't match! Index = {} num_slices = {}",
        args.slice_index,
        args.num_slices);
    TT_FATAL(input_tensor.layout() == Layout::TILE, "Currently, only tile layout is supported for partial S->I");
    TT_FATAL(
        (cache_tensor.physical_volume() / cache_tensor.padded_shape()[-1]) % args.num_slices == 0,
        "Total height of a tensor must be divisible by num_slices!");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");

    TT_FATAL(input_tensor.memory_config().is_sharded(), "Input tensor must be sharded");
    if (input_tensor.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        if (input_tensor.padded_shape()[-1] % shard_spec.shape[1] != 0 ||
            ((input_tensor.physical_volume() / input_tensor.padded_shape()[-1]) % shard_spec.shape[0]) != 0) {
            TT_FATAL(
                input_tensor.shard_spec().value().grid.ranges().size() == 1,
                "Input tensor shard spec must have exactly 1 grid range but got {}",
                input_tensor.shard_spec().value().grid.ranges().size());
        }
    }
    if (input_tensor.dtype() != args.output_dtype) {
        TT_FATAL(
            input_tensor.layout() == Layout::TILE,
            "Input tensor layout must be TILE but got {}",
            input_tensor.layout());
    }
}

void ShardedToInterleavedPartialDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

TensorSpec ShardedToInterleavedPartialDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // Return the spec of the cache tensor since we're writing to it
    return tensor_args.cache_tensor.tensor_spec();
}

Tensor ShardedToInterleavedPartialDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // Return the cache tensor itself - it's preallocated
    return tensor_args.cache_tensor;
}

tt::tt_metal::operation::OpPerformanceModelGeneral<ShardedToInterleavedPartialDeviceOperation::tensor_return_value_t>
ShardedToInterleavedPartialDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) const {
    int ideal_dev_clock_cycles =
        ttnn::operations::data_movement::common_tm_bw_model(tensor_args.input_tensor, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {tensor_args.input_tensor}, {output_tensor}, ideal_dev_clock_cycles);
    return result;
}

Tensor sharded_to_interleaved_partial(
    const Tensor& input_tensor,
    const Tensor& cache_tensor,
    uint32_t num_slices,
    uint32_t slice_index,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype) {
    return ttnn::device_operation::launch<ShardedToInterleavedPartialDeviceOperation>(
        ShardedToInterleavedPartialParams{
            .num_slices = num_slices,
            .slice_index = slice_index,
            .output_mem_config = output_mem_config,
            .output_dtype = output_dtype},
        ShardedToInterleavedPartialInputs{input_tensor, cache_tensor});
}

}  // namespace ttnn::prim
