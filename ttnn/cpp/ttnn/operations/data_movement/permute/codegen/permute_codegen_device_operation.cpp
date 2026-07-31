// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>

#include "permute_codegen_device_operation.hpp"
#include "permute_codegen_supported.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::data_movement {

static std::array<uint32_t, PermuteCodegenDeviceOperation::kMaxDims> get_row_strides(const ttnn::Shape& shape) {
    std::array<uint32_t, PermuteCodegenDeviceOperation::kMaxDims> strides{};
    const uint32_t rank = shape.rank();
    strides[rank - 1] = 1;
    strides[rank - 2] = 1;
    for (int i = static_cast<int>(rank) - 3; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

PermuteCodegenDeviceOperation::program_factory_t PermuteCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    // dims[-1] == rank - 1: last dim unchanged -> row-invariant. Otherwise -> blocked-generic.
    if (operation_attributes.dims[operation_attributes.rank - 1] == operation_attributes.rank - 1) {
        return MultiCoreRowInvariant{};
    }
    return MultiCoreBlockedGeneric{};
}

void PermuteCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        attributes.rank == tensor_args.input_tensor.logical_shape().rank(),
        "PermuteCodegen: permutation rank must match input tensor rank");
    TT_FATAL(
        tensor_args.input_tensor.layout() == Layout::ROW_MAJOR,
        "PermuteCodegen: only ROW_MAJOR is supported by the codegen port");
    ttsl::SmallVector<uint32_t> dims(attributes.dims.begin(), attributes.dims.begin() + attributes.rank);
    TT_FATAL(
        supported_by_codegen(tensor_args.input_tensor, dims),
        "PermuteCodegen: call does not satisfy supported_by_codegen()");
}

void PermuteCodegenDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& /*tensor_args*/) {}

PermuteCodegenDeviceOperation::spec_return_value_t PermuteCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.logical_shape();

    ttsl::SmallVector<uint32_t> output_shape_vec(attributes.rank);
    std::transform(
        attributes.dims.begin(),
        attributes.dims.begin() + attributes.rank,
        output_shape_vec.begin(),
        [&](uint32_t dim) { return input_shape[dim]; });
    auto output_shape = Shape(std::move(output_shape_vec));

    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), attributes.output_mem_config));
}

PermuteCodegenDeviceOperation::tensor_return_value_t PermuteCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::PermuteCodegenDeviceOperation::tensor_return_value_t permute_codegen(
    const Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    using OperationType = ttnn::operations::data_movement::PermuteCodegenDeviceOperation;

    const auto& input_shape = input_tensor.logical_shape();
    const uint32_t rank = input_shape.rank();

    std::array<uint32_t, OperationType::kMaxDims> padded_dims{};
    std::array<uint32_t, OperationType::kMaxDims> padded_input_shape{};
    for (uint32_t i = 0; i < rank; i++) {
        padded_dims[i] = dims[i];
        padded_input_shape[i] = input_shape[i];
    }

    ttsl::SmallVector<uint32_t> output_shape_vec(rank);
    std::transform(dims.begin(), dims.end(), output_shape_vec.begin(), [&](uint32_t dim) { return input_shape[dim]; });
    auto output_shape = ttnn::Shape(std::move(output_shape_vec));

    // aligned_stick_bytes and num_blocks_total are branch-specific (row-invariant vs.
    // blocked-generic geometry per permute.yaml) and are computed by the program factory in 4a.
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .rank = rank,
            .dims = padded_dims,
            .input_shape = padded_input_shape,
            .output_strides = ttnn::operations::data_movement::get_row_strides(output_shape),
            .num_rows = static_cast<uint32_t>(output_shape.volume() / output_shape[-1]),
            .aligned_stick_bytes = 0,
            .elem_size = input_tensor.element_size(),
            .num_blocks_total = 0,
            .output_mem_config = memory_config.value_or(input_tensor.memory_config())},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor, .optional_output_tensor = std::move(optional_output_tensor)});
}
}  // namespace ttnn::prim
