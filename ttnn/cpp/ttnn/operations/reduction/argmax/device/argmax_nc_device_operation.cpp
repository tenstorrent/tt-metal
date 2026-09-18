// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "argmax_nc_device_operation.hpp"
#include "argmax_utils.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

void ArgMaxNCDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    TT_FATAL(
        input.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "argmax_nc: only INTERLEAVED input memory layout is supported, got {}",
        input.memory_config().memory_layout());

    TT_FATAL(
        args.output_mem_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "argmax_nc: only INTERLEAVED output memory layout is supported, got {}",
        args.output_mem_config.memory_layout());

    TT_FATAL(
        input.layout() == tt::tt_metal::Layout::TILE,
        "argmax_nc: input must be in TILE layout, got {}",
        input.layout());

    TT_FATAL(
        input.dtype() == tt::tt_metal::DataType::BFLOAT16 || input.dtype() == tt::tt_metal::DataType::FLOAT32,
        "argmax_nc: only BFLOAT16 or FLOAT32 inputs are supported, got {}",
        input.dtype());

    const auto& input_shape = input.padded_shape();
    const int32_t rank = static_cast<int32_t>(input_shape.rank());
    TT_FATAL(rank >= 2, "argmax_nc: input rank must be >= 2, got {}", rank);

    const int32_t normalized_dim = normalize_dim(args.dim, rank);
    TT_FATAL(
        normalized_dim >= 0 && normalized_dim < rank - 2,
        "argmax_nc: reduction dim (normalized={}) must be a non-HW dim, i.e. in [0, {}).",
        normalized_dim,
        rank - 2);

    if (tensor_args.preallocated_output.has_value()) {
        const auto& out = tensor_args.preallocated_output.value();
        TT_FATAL(
            out.dtype() == tt::tt_metal::DataType::UINT32,
            "argmax_nc: preallocated output must be UINT32, got {}",
            out.dtype());
        TT_FATAL(
            out.layout() == tt::tt_metal::Layout::TILE,
            "argmax_nc: preallocated output must be TILE layout, got {}",
            out.layout());
    }
}

tt::tt_metal::TensorSpec ArgMaxNCDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto& input = tensor_args.input;
    const auto& input_shape = input.logical_shape();
    const int32_t rank = static_cast<int32_t>(input_shape.rank());
    const int32_t normalized_dim = normalize_dim(args.dim, rank);

    // Output logical shape: same as input logical shape but with the reduced
    // dim set to 1 (keepdim semantics). TensorLayout(TILE) will pad the
    // inner-2 dims to tile dims internally. The host-facing `argmax` wrapper
    // is responsible for untilizing / squeezing / slicing to match the
    // user-visible contract.
    auto output_shape = input_shape;
    output_shape[normalized_dim] = 1;

    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            args.output_mem_config));
}

Tensor ArgMaxNCDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

Tensor argmax_nc(
    const Tensor& input,
    int32_t dim,
    const std::optional<Tensor>& preallocated_output,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType = ArgMaxNCDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .dim = dim,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
        .sub_core_grids = sub_core_grids,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .input = input,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
