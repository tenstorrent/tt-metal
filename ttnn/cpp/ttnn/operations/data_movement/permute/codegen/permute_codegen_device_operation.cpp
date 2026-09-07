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

namespace {

// A caller-supplied output is adopted verbatim by compute_output_specs, so nothing downstream
// re-derives what the kernels were sized against. Run from both validators, not just the miss one:
// the program hash covers tensor specs only -- DeviceStorage contributes an empty attribute tuple --
// so an output with the right spec sitting on a different device hashes identically to one on the
// input's device and would reach a cached program as a hit.
void validate_optional_output(
    const PermuteCodegenDeviceOperation::operation_attributes_t& attributes,
    const PermuteCodegenDeviceOperation::tensor_args_t& tensor_args) {
    if (!tensor_args.optional_output_tensor.has_value()) {
        return;
    }
    const auto& output_tensor = *tensor_args.optional_output_tensor;
    TT_FATAL(
        output_tensor.storage_type() == StorageType::DEVICE,
        "PermuteCodegen: preallocated output tensor must be on device");
    TT_FATAL(output_tensor.buffer() != nullptr, "PermuteCodegen: preallocated output tensor has no allocated buffer");
    TT_FATAL(
        output_tensor.device() == tensor_args.input_tensor.device(),
        "PermuteCodegen: preallocated output tensor must live on the same device as the input");
    TT_FATAL(
        output_tensor.tensor_spec() ==
            PermuteCodegenDeviceOperation::compute_output_specs(attributes, {tensor_args.input_tensor, std::nullopt}),
        "PermuteCodegen: preallocated output tensor spec does not match the spec this permute produces");
}

}  // namespace

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
    if (permute_codegen::is_row_invariant({operation_attributes.dims.data(), operation_attributes.rank})) {
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
    // supported_by_codegen() answers over layout, dtype, shape and memory config, all of which a
    // host tensor answers too, so the structural preconditions are asserted separately -- otherwise
    // a host tensor reaching the prim directly reads as an out-of-scope shape.
    TT_FATAL(
        tensor_args.input_tensor.storage_type() == StorageType::DEVICE,
        "PermuteCodegen: input tensor must be on device");
    TT_FATAL(tensor_args.input_tensor.buffer() != nullptr, "PermuteCodegen: input tensor has no allocated buffer");
    ttsl::SmallVector<uint32_t> dims(attributes.dims.begin(), attributes.dims.begin() + attributes.rank);
    TT_FATAL(
        permute_codegen::supported_by_codegen(tensor_args.input_tensor, dims, attributes.output_mem_config),
        "PermuteCodegen: call does not satisfy supported_by_codegen()");
    validate_optional_output(attributes, tensor_args);
}

void PermuteCodegenDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_optional_output(attributes, tensor_args);
}

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

    // supported_by_codegen() admits ROW_MAJOR only, and a row-major page config carries no tile at
    // all, so building it from the layout drops nothing: input and output are paged identically.
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

    // Asserted here rather than in validate, which the framework runs only after this function has
    // already filled rank-indexed fixed-size arrays and divided by the trailing extent. A call
    // arriving through ttnn::permute or permute_force_codegen has cleared supported_by_codegen()
    // and satisfies all of this; a direct prim call has not.
    TT_FATAL(
        rank >= 2 && rank <= OperationType::kMaxDims,
        "prim::permute_codegen: rank must be between 2 and {}, got {}",
        OperationType::kMaxDims,
        rank);
    TT_FATAL(
        dims.size() == rank,
        "prim::permute_codegen: permutation has {} entries for a rank-{} input",
        dims.size(),
        rank);
    TT_FATAL(
        ttnn::operations::data_movement::permute_codegen::is_permutation(dims),
        "prim::permute_codegen: dims must be a permutation of [0, {}) with no repeated axis",
        rank);
    for (uint32_t i = 0; i < rank; i++) {
        TT_FATAL(input_shape[i] > 0, "prim::permute_codegen: input dim {} is zero", i);
    }

    std::array<uint32_t, OperationType::kMaxDims> padded_dims{};
    std::array<uint32_t, OperationType::kMaxDims> padded_input_shape{};
    for (uint32_t i = 0; i < rank; i++) {
        padded_dims[i] = dims[i];
        padded_input_shape[i] = input_shape[i];
    }

    ttsl::SmallVector<uint32_t> output_shape_vec(rank);
    std::transform(dims.begin(), dims.end(), output_shape_vec.begin(), [&](uint32_t dim) { return input_shape[dim]; });
    auto output_shape = ttnn::Shape(std::move(output_shape_vec));

    const auto output_mem_config = memory_config.value_or(input_tensor.memory_config());

    // Total rows (=volume/W) uses the INPUT's last dim on both branches: the blocked-generic
    // kernels count input rows, which differs from volume/output_shape[-1] (== X) on the W-changing
    // branch. Row-invariant permutes keep input W == output W, so this is the same count there.
    const uint32_t num_rows = static_cast<uint32_t>(input_shape.volume() / input_shape[rank - 1]);

    uint32_t num_blocks_total = 0;
    if (!ttnn::operations::data_movement::permute_codegen::is_row_invariant(dims)) {
        // W-changing (blocked-generic) block count.
        constexpr uint32_t kBlockSize = 32;
        const uint32_t x_dim = dims[rank - 1];
        const uint32_t X = input_shape[x_dim];
        const uint32_t x_blocks = (X + kBlockSize - 1) / kBlockSize;
        const uint32_t w_blocks = (input_shape[rank - 1] + kBlockSize - 1) / kBlockSize;
        const uint32_t non_x_rows = num_rows / X;
        num_blocks_total = non_x_rows * x_blocks * w_blocks;
    }

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .rank = rank,
            .dims = padded_dims,
            .input_shape = padded_input_shape,
            .output_strides = ttnn::operations::data_movement::get_row_strides(output_shape),
            .num_rows = num_rows,
            .elem_size = input_tensor.element_size(),
            .num_blocks_total = num_blocks_total,
            .output_mem_config = output_mem_config},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor, .optional_output_tensor = std::move(optional_output_tensor)});
}
}  // namespace ttnn::prim
