// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_write_device_operation.hpp"

#include <tt_stl/assert.hpp>
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

SliceWriteDeviceOperation::program_factory_t SliceWriteDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    bool has_step = false;
    for (unsigned int step_val : operation_attributes.step) {
        if (step_val != 1) {
            has_step = true;
            break;
        }
    }

    // Logic from slice_write_multi_core
    if (input.is_sharded()) {
        TT_FATAL(!has_step, "Step is not supported for sharded slice_write operation");
        if (input.layout() == Layout::ROW_MAJOR) {
            return SliceWriteRMShardedInputProgramFactory{};
        }
        if (input.layout() == Layout::TILE) {
            return SliceWriteTiledShardedInputProgramFactory{};
        }
        TT_THROW("Unsupported input memory layout for slice_write operation");

    } else {
        return SliceWriteRMInterleavedProgramFactory{};
    }
}

void SliceWriteDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void SliceWriteDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& output_tensor = tensor_args.output;
    const auto output_padded_shape = output_tensor.padded_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to slice_write need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to slice_write need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.layout() == Layout::TILE || input_tensor.layout() == Layout::ROW_MAJOR,
        "Input tensor layout must be TILE or ROW_MAJOR but got {}",
        input_tensor.layout());
    TT_FATAL(
        input_tensor.padded_shape().rank() == args.slice_start.rank() &&
            output_padded_shape.rank() == args.slice_start.rank() && args.slice_start.rank() == args.slice_end.rank(),
        "Ranks of input tensor, output_tensor, slice start and slice end should be equal. Got {} {} {} {}",
        input_tensor.padded_shape().rank(),
        output_padded_shape.rank(),
        args.slice_start.rank(),
        args.slice_end.rank());
    for (uint32_t i = 0; i < output_padded_shape.rank(); i++) {
        TT_FATAL(
            args.slice_start[i] < output_padded_shape[i],
            "Start is outside the bounds of the output tensor for index {}. Got {}. Size {}",
            i,
            args.slice_start[i],
            output_padded_shape[i]);
        TT_FATAL(
            args.slice_end[i] <= output_padded_shape[i],
            "Ends {} must be less than or equal to the shape of the tensor {}",
            args.slice_end[i],
            output_padded_shape[i]);
        // Check if start shape is <= end shape
        TT_FATAL(
            args.slice_start[i] <= args.slice_end[i],
            "Slice start {} should be less than slice end {}",
            args.slice_start[i],
            args.slice_end[i]);
    }
    // If the input tensor is sharded, then rank should be 4
    TT_FATAL(
        !input_tensor.is_sharded() || input_tensor.padded_shape().rank() == 4,
        "Sharded input tensor should be of rank 4. Got {}",
        input_tensor.padded_shape().rank());
}

TensorSpec SliceWriteDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.output.tensor_spec();
}

tt::stl::hash::hash_t SliceWriteDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "SliceWriteDeviceOperation::compute_program_hash is called");

    auto program_factory = select_program_factory(args, tensor_args);

    return tt::tt_metal::operation::hash_operation<SliceWriteDeviceOperation>(
        args, tensor_args, program_factory.index());
}

Tensor SliceWriteDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.output;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor slice_write(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& slice_start,
    const ttnn::Shape& slice_end,
    const ttnn::Shape& step) {
    using OperationType = ttnn::experimental::prim::SliceWriteDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .slice_start = slice_start,
        .slice_end = slice_end,
        .step = step,
    };
    auto tensor_args = OperationType::tensor_args_t{.input = input_tensor, .output = output_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
