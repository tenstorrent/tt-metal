// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pow_int_device_operation.hpp"

#include <algorithm>
#include <array>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace CMAKE_UNIQUE_NAMESPACE {

constexpr std::array kSupportedDtypes{DataType::INT32, DataType::UINT32, DataType::UINT16};

bool is_supported_dtype(DataType dtype) { return std::ranges::find(kSupportedDtypes, dtype) != kSupportedDtypes.end(); }

}  // namespace CMAKE_UNIQUE_NAMESPACE

void PowIntDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    TT_FATAL(
        input.storage_type() == StorageType::DEVICE,
        "pow_int: input must be on device, got storage type {}",
        input.storage_type());
    TT_FATAL(input.buffer() != nullptr, "pow_int: input must be allocated in a device buffer");
    TT_FATAL(input.layout() == Layout::TILE, "pow_int: input must be in TILE layout, got {}", input.layout());
    TT_FATAL(
        CMAKE_UNIQUE_NAMESPACE::is_supported_dtype(input.dtype()),
        "pow_int: input dtype must be INT32, UINT32 or UINT16, got {}",
        input.dtype());

    if (tensor_args.preallocated_output.has_value()) {
        const auto& output = *tensor_args.preallocated_output;
        TT_FATAL(
            output.dtype() == input.dtype(),
            "pow_int: output dtype {} must match input dtype {}",
            output.dtype(),
            input.dtype());
        TT_FATAL(output.layout() == Layout::TILE, "pow_int: output must be in TILE layout, got {}", output.layout());
        TT_FATAL(
            output.logical_shape() == input.logical_shape(),
            "pow_int: output shape {} must match input shape {}",
            output.logical_shape(),
            input.logical_shape());
    }
}

PowIntDeviceOperation::spec_return_value_t PowIntDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    const auto& input = tensor_args.input;
    return TensorSpec(input.logical_shape(), TensorLayout(input.dtype(), Layout::TILE, args.output_memory_config));
}

PowIntDeviceOperation::tensor_return_value_t PowIntDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

bool PowIntDeviceOperation::skip_launch(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/, const tensor_return_value_t& output) {
    return output.logical_shape().volume() == 0;
}

Tensor pow_int(
    const Tensor& input,
    uint32_t exponent,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& preallocated_output) {
    return ttnn::device_operation::launch<PowIntDeviceOperation>(
        PowIntParams{.exponent = exponent, .output_memory_config = output_memory_config},
        PowIntInputs{.input = input, .preallocated_output = preallocated_output});
}

}  // namespace ttnn::prim
