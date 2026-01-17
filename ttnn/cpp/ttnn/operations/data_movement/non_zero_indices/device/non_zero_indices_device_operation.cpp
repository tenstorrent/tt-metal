// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_device_operation_types.hpp"
#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_program_factory.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

NonZeroIndicesDeviceOperation::program_factory_t NonZeroIndicesDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return NonZeroIndicesProgramFactory{};
}

void NonZeroIndicesDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void NonZeroIndicesDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    auto input_tensor_shape = input_tensor.padded_shape();
    TT_FATAL(
        input_tensor_shape[0] == 1 and input_tensor_shape[1] == 1 and input_tensor_shape[2] == 1,
        "The input shape must be 4D with the following form: 1, 1, 1, X.");
    TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR, "Currently only supporting row major layout");
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to Non-zero need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to Non-zero need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Non-zero does not currently support sharding");
}

NonzeroResultSpec NonZeroIndicesDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    ttnn::Shape num_non_zero_shape({1, 1, 1, 8});
    TensorLayout layout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), args.output_memory_config);
    return {TensorSpec(num_non_zero_shape, layout), TensorSpec(tensor_args.input.logical_shape(), layout)};
}

NonzeroResult NonZeroIndicesDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto output_specs = compute_output_specs(args, tensor_args);
    return {
        create_device_tensor(std::get<0>(output_specs), tensor_args.input.device()),
        create_device_tensor(std::get<1>(output_specs), tensor_args.input.device()),
    };
}

NonzeroResult nonzero(const Tensor& input_tensor, const tt::tt_metal::MemoryConfig& memory_config) {
    return ttnn::device_operation::launch<NonZeroIndicesDeviceOperation>(
        NonzeroParams{.output_memory_config = memory_config}, NonzeroInputs{.input = input_tensor});
}

}  // namespace ttnn::prim
