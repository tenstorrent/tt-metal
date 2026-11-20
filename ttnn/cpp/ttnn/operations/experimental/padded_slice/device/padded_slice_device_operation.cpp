// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "padded_slice_device_operation.hpp"

#include <cmath>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {

// pad_value is a float, but an integer output takes it through a float to integer cast, which is undefined for a
// fractional, non-finite or out-of-range value. The bounds are powers of two, which a float holds exactly.
bool pad_value_fits_output_dtype(float value, DataType dtype) {
    switch (dtype) {
        case DataType::INT32:
            return std::isfinite(value) && std::trunc(value) == value && value >= -2147483648.0f &&
                   value < 2147483648.0f;
        case DataType::UINT32:
            return std::isfinite(value) && std::trunc(value) == value && value >= 0.0f && value < 4294967296.0f;
        case DataType::UINT16:
            return std::isfinite(value) && std::trunc(value) == value && value >= 0.0f && value < 65536.0f;
        case DataType::UINT8:
            return std::isfinite(value) && std::trunc(value) == value && value >= 0.0f && value < 256.0f;
        default: return true;
    }
}

}  // namespace

PaddedSliceDeviceOperation::program_factory_t PaddedSliceDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    if (tensor_args.input.layout() == Layout::ROW_MAJOR) {
        return PaddedSliceRMProgramFactory{};
    }
    if (tensor_args.input.layout() == Layout::TILE) {
        return PaddedSliceTileProgramFactory{};
    }
    TT_THROW("Unsupported layout for padded_slice operation: {}", tensor_args.input.layout());
}

void PaddedSliceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input;

    // Validate step parameter early - padded_slice does not support strided slices
    const bool has_step = std::any_of(args.step.cbegin(), args.step.cend(), [](uint32_t s) { return s != 1; });
    TT_FATAL(!has_step, "Padded slice does not support strided slices");

    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to unpad need to be on device!");
    TT_FATAL(input_tensor_a.is_allocated(), "Operands to unpad need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.padded_shape().rank() == 4, "Only 4D tensors are supported for padded_slice");
    TT_FATAL(
        input_tensor_a.padded_shape().rank() == args.padded_slice_start.rank() &&
            args.padded_slice_start.rank() == args.padded_slice_end.rank(),
        "Padded slice start, end and input tensor must all have the same rank");
    for (uint32_t i = 0; i < input_tensor_a.padded_shape().rank(); i++) {
        TT_FATAL(
            args.padded_slice_start[i] < input_tensor_a.padded_shape()[i],
            "Starts {} must be less than the shape of the tensor {} at index {}",
            args.padded_slice_start[i],
            input_tensor_a.padded_shape()[i],
            i);
        TT_FATAL(
            args.padded_slice_end[i] <= input_tensor_a.padded_shape()[i],
            "Ends {} must be less than or equal to the shape of the tensor {}",
            args.padded_slice_end[i],
            input_tensor_a.padded_shape()[i]);
        // Check if start shape is <= end shape
        TT_FATAL(
            args.padded_slice_start[i] <= args.padded_slice_end[i],
            "Slice start {} must be less than or equal to the end {}",
            args.padded_slice_start[i],
            args.padded_slice_end[i]);
    }
    if (args.pad_value.has_value()) {
        const auto output_dtype = tensor_args.preallocated_output.has_value()
                                      ? tensor_args.preallocated_output->dtype()
                                      : compute_output_specs(args, tensor_args).data_type();
        TT_FATAL(
            pad_value_fits_output_dtype(*args.pad_value, output_dtype),
            "pad_value {} must be a finite integer that fits the {} output dtype",
            *args.pad_value,
            output_dtype);
    }
    if (tensor_args.preallocated_output.has_value()) {
        const auto output_shape_required = compute_output_specs(args, tensor_args).padded_shape();
        const auto& out_tensor = tensor_args.preallocated_output.value();
        TT_FATAL(
            out_tensor.padded_shape() == output_shape_required,
            "The input tensors need a shape of {}, however the output tensor is only {}",
            output_shape_required,
            out_tensor.padded_shape());
    }
}

tt::tt_metal::TensorSpec PaddedSliceDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    ttsl::SmallVector<uint32_t> out_shape(input_tensor.logical_shape().rank());

    TT_FATAL(out_shape.size() == 4, "Only 4D tensors are supported for padded_slice");
    auto output_dim_i = [&args](size_t i) {
        return (args.padded_slice_end[i] - args.padded_slice_start[i] + args.step[i] - 1) / args.step[i];
    };
    for (uint32_t i = 0; i < out_shape.size(); i++) {
        out_shape[i] = output_dim_i(i);
    }
    out_shape[2] = out_shape[0] * out_shape[1] * out_shape[2];
    out_shape[0] = 1;
    out_shape[1] = 1;

    if (args.output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        auto output_shard_shape = args.output_mem_config.shard_spec().value().shape;
        out_shape[out_shape.size() - 1] = output_shard_shape[1];
    } else if (args.output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        auto output_shard_shape = args.output_mem_config.shard_spec().value().shape;
        out_shape[out_shape.size() - 2] = output_shard_shape[0];
    }

    ttnn::Shape output_tensor_shape(std::move(out_shape));
    auto output_dtype = input_tensor.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.dtype();
    auto tensor_layout = TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), args.output_mem_config);
    return tt::tt_metal::TensorSpec(output_tensor_shape, tensor_layout);
}

Tensor PaddedSliceDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::experimental::prim::PaddedSliceDeviceOperation::tensor_return_value_t padded_slice(
    const Tensor& input,
    const ttnn::Shape& padded_slice_start,
    const ttnn::Shape& padded_slice_end,
    const ttnn::Shape& step,
    const MemoryConfig& output_mem_config,
    const std::optional<Tensor>& preallocated_output,
    const std::optional<float>& pad_value) {
    using OperationType = ttnn::experimental::prim::PaddedSliceDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .padded_slice_start = padded_slice_start,
        .padded_slice_end = padded_slice_end,
        .step = step,
        .output_mem_config = output_mem_config,
        .pad_value = pad_value};
    auto tensor_args = OperationType::tensor_args_t{.input = input, .preallocated_output = preallocated_output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
