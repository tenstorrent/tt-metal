// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/split/device/split_device_operation.hpp"
#include "ttnn/operations/data_movement/split/device/split_program_factory.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::split {

void SplitDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    TT_FATAL(args.dim == 3 || args.dim == 2, "Split is possible along dim 2 or 3 only");
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Split does not currently support sharding");
    TT_FATAL(
        args.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Split does not currently support sharding");

    TT_FATAL(input_tensor.padded_shape()[0] == 1, "shape[0] must be 1 (batch 1 only)");
    TT_FATAL(
        input_tensor.padded_shape()[args.dim] % args.num_splits == 0,
        "Dim being split must be evenly divisible by number of splits");
    TT_FATAL(
        args.dim <= input_tensor.padded_shape().rank() && args.dim >= 0, "Dim being split must be from 0 to rank - 1");
    TT_FATAL(input_tensor.padded_shape().rank() == 4, "Tensor needs to be rank 4");
    TT_FATAL(input_tensor.layout() == Layout::TILE, "Tensor needs to be in TILE Layout");
}

void SplitDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

SplitDeviceOperation::spec_return_value_t SplitDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    auto input_shape_array = input_tensor.padded_shape().to_array_4D();
    auto output_shape_array = input_shape_array;
    output_shape_array[args.dim] /= args.num_splits;
    TensorSpec spec(
        Shape(output_shape_array),
        TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.layout()), args.output_mem_config));
    return std::vector<ttnn::TensorSpec>(args.num_splits, spec);
}

SplitDeviceOperation::tensor_return_value_t SplitDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    auto output_specs = compute_output_specs(args, tensor_args);

    std::vector<Tensor> output_tensors;
    output_tensors.reserve(args.num_splits);
    for (const auto& spec : output_specs) {
        output_tensors.push_back(create_device_tensor(spec, input_tensor.device()));
    }
    return output_tensors;
}

SplitDeviceOperation::program_factory_t SplitDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return program::SplitProgramFactory{};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<SplitDeviceOperation::tensor_return_value_t>
SplitDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors) {
    const auto& input_tensor = tensor_args.input;
    std::vector<Tensor> input_tensors = {input_tensor};

    // Calculate ideal device clock cycles using the actual output tensor
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output_tensors.at(0), false, 0, false, true);

    return tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t>(
        input_tensors, output_tensors, ideal_dev_clock_cycles);
}

}  // namespace ttnn::operations::data_movement::split

namespace ttnn::prim {
std::vector<ttnn::Tensor> split(
    const Tensor& input_tensor, int num_splits, int dim, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = ttnn::operations::data_movement::split::SplitDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{num_splits, dim, output_mem_config},
        OperationType::tensor_args_t{input_tensor});
}
}  // namespace ttnn::prim
