// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_codegen_device_operation.hpp"

#include <algorithm>
#include <array>

#include <tt-metalium/tt_align.hpp>

#include "permute_codegen_supported.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::data_movement {

PermuteCodegenDeviceOperation::program_factory_t PermuteCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    // dims[-1] == rank - 1: last dim unchanged, row-invariant no-compute path.
    if (operation_attributes.dims[operation_attributes.rank - 1] == operation_attributes.rank - 1) {
        return RowInvariant{};
    }
    return BlockedGeneric{};
}

void PermuteCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        permute_codegen::supported_by_codegen(
            tensor_args.input_tensor, ttsl::Span<const uint32_t>(attributes.dims.data(), attributes.rank)),
        "PermuteCodegenDeviceOperation: input is not supported by the codegen implementation");
}

void PermuteCodegenDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& /*tensor_args*/) {}

PermuteCodegenDeviceOperation::spec_return_value_t PermuteCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    const auto input_shape = input_tensor.logical_shape();

    ttsl::SmallVector<uint32_t> output_shape_vec(attributes.rank);
    for (uint32_t i = 0; i < attributes.rank; ++i) {
        output_shape_vec[i] = input_shape[attributes.dims[i]];
    }

    return TensorSpec(
        Shape(std::move(output_shape_vec)),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), attributes.output_mem_config));
}

PermuteCodegenDeviceOperation::tensor_return_value_t PermuteCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

ttsl::hash::hash_t PermuteCodegenDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    const auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return tt::tt_metal::operation::hash_operation<PermuteCodegenDeviceOperation>(
        operation_attributes.rank,
        operation_attributes.dims,
        operation_attributes.input_shape,
        operation_attributes.output_strides,
        operation_attributes.num_rows,
        operation_attributes.aligned_stick_bytes,
        operation_attributes.elem_size,
        operation_attributes.num_blocks_total,
        operation_attributes.output_mem_config,
        program_factory.index(),
        input_tensor.tensor_spec(),
        input_tensor.padded_shape(),
        output_spec,
        output_spec.padded_shape());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<PermuteCodegenDeviceOperation::tensor_return_value_t>
PermuteCodegenDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*op_attr*/, const tensor_args_t& inputs, const Tensor& output) {
    const auto& input_tensor = inputs.input_tensor;
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output, false, 0, true);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::PermuteCodegenDeviceOperation::tensor_return_value_t permute_codegen(
    const Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    using ttnn::operations::data_movement::PermuteCodegenDeviceOperation;
    constexpr uint32_t kMaxDims = PermuteCodegenDeviceOperation::MAX_DIMS;

    const auto input_shape = input_tensor.logical_shape();
    const uint32_t rank = input_shape.rank();
    TT_FATAL(dims.size() == rank, "permute_codegen: dims length {} does not match tensor rank {}", dims.size(), rank);
    // Structural guard: operation_attributes_t packs dims/shape/strides into fixed-size
    // std::array<uint32_t, MAX_DIMS>; supported_by_codegen() rejects this before routing ever
    // reaches here, but a direct forced-codegen call must not overrun the arrays below.
    TT_FATAL(rank <= kMaxDims, "permute_codegen: rank {} exceeds the maximum supported rank {}", rank, kMaxDims);

    std::array<uint32_t, kMaxDims> dims_arr{};
    std::array<uint32_t, kMaxDims> input_shape_arr{};
    for (uint32_t i = 0; i < rank; ++i) {
        dims_arr[i] = dims[i];
        input_shape_arr[i] = input_shape[i];
    }

    ttsl::SmallVector<uint32_t> output_shape_vec(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        output_shape_vec[i] = input_shape[dims[i]];
    }

    // Row-unit strides over all but the last (W) dim, ported from an internal reference
    // implementation's get_row_strides(output_shape). The last two slots are both stride 1 — W
    // is intra-row, and rank-2 is the innermost row-counted dim — matched exactly by both
    // writers' addressing math.
    std::array<uint32_t, kMaxDims> output_strides{};
    if (rank == 1) {
        output_strides[0] = 1;
    } else {
        output_strides[rank - 1] = 1;
        output_strides[rank - 2] = 1;
        for (int32_t i = static_cast<int32_t>(rank) - 3; i >= 0; --i) {
            output_strides[i] = output_strides[i + 1] * output_shape_vec[i + 1];
        }
    }

    const uint32_t elem_size = input_tensor.element_size();
    const uint32_t w = input_shape[rank - 1];
    uint32_t num_rows = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        num_rows *= input_shape[i];
    }
    num_rows /= w;

    const uint32_t stick_bytes = w * elem_size;
    const uint32_t aligned_stick_bytes = tt::align(stick_bytes, input_tensor.buffer()->alignment());

    uint32_t num_blocks_total = 0;
    if (dims[rank - 1] != rank - 1) {
        // W-changing (BlockedGeneric): 32x32 block count over the permuted-last (x_dim) and W
        // axes, matching build_permute_rm_blocked's host section exactly.
        constexpr uint32_t kBlock = 32;
        const uint32_t x_dim = dims[rank - 1];
        const uint32_t x = input_shape[x_dim];
        const uint32_t x_blocks = (x + kBlock - 1) / kBlock;
        const uint32_t w_blocks = (w + kBlock - 1) / kBlock;
        const uint32_t non_x_rows = num_rows / x;
        num_blocks_total = non_x_rows * x_blocks * w_blocks;
    }

    PermuteCodegenDeviceOperation::operation_attributes_t attrs{
        .rank = rank,
        .dims = dims_arr,
        .input_shape = input_shape_arr,
        .output_strides = output_strides,
        .num_rows = num_rows,
        .aligned_stick_bytes = aligned_stick_bytes,
        .elem_size = elem_size,
        .num_blocks_total = num_blocks_total,
        .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
    };

    return ttnn::device_operation::launch<PermuteCodegenDeviceOperation>(
        attrs, PermuteCodegenDeviceOperation::tensor_args_t{input_tensor, std::move(optional_output_tensor)});
}
}  // namespace ttnn::prim
