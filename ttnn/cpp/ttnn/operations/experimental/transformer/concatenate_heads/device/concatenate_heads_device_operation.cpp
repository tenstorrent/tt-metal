// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concatenate_heads_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "concatenate_heads_program_factory.hpp"
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

ConcatenateHeadsDeviceOperation::program_factory_t ConcatenateHeadsDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return ConcatenateHeadsProgramFactory{};
}

void ConcatenateHeadsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void ConcatenateHeadsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& preallocated_output = tensor_args.preallocated_output;

    const auto batch_size = input_tensor.padded_shape()[0];
    // TODO: See issue #1744
    TT_FATAL(batch_size >= 7 && batch_size <= 9, "Input batch size must be between 7 to 9 for bert large TM ops!");

    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");

    TT_FATAL((input_tensor.padded_shape() == ttnn::Shape({batch_size, 16, 384, 64})), "Unsupported input shape");

    if (preallocated_output.has_value()) {
        TT_FATAL(
            preallocated_output.value().dtype() == input_tensor.dtype(), "Output dtype must be same as input dtype!");

        TT_FATAL(
            preallocated_output.value().padded_shape() == ttnn::Shape({batch_size, 1, 384, 1024}),
            "Output shape must be (batch_size, 1, 384, 1024)!");
    }

    auto device_compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    TT_FATAL(
        (args.compute_with_storage_grid_size.x <= device_compute_with_storage_grid_size.x &&
         args.compute_with_storage_grid_size.y <= device_compute_with_storage_grid_size.y),
        "Unsupported grid shape");
}

TensorSpec ConcatenateHeadsDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input;
    const auto batch_size = input_tensor.padded_shape()[0];
    ttnn::Shape output_shape({batch_size, 1, 384, 1024});
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), args.output_mem_config));
}

Tensor ConcatenateHeadsDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }

    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t ConcatenateHeadsDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.padded_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    operation::Hash hash = operation::hash_operation<ConcatenateHeadsDeviceOperation>(
        args.output_mem_config,
        program_factory.index(),
        input_tensor.dtype(),
        input_tensor.memory_config(),
        input_shape.volume());

    return hash;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::ConcatenateHeadsDeviceOperation::tensor_return_value_t concatenate_heads(
    const ttnn::Tensor& input_tensor,
    const CoreCoord& compute_with_storage_grid_size,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = ttnn::experimental::prim::ConcatenateHeadsDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .compute_with_storage_grid_size = compute_with_storage_grid_size,
        .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
    };
    auto tensor_args = OperationType::tensor_args_t{.input = input_tensor, .preallocated_output = preallocated_output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
