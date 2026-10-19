// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/permute/codegen/permute_codegen_device_operation.hpp"

#include <algorithm>
#include <utility>

#include <tt_stl/assert.hpp>
#include <tt_stl/small_vector.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/permute/codegen/permute_codegen_supported.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

namespace {

ttsl::SmallVector<uint32_t> active_dims(const PermuteCodegenParams& params, uint32_t rank) {
    TT_FATAL(
        rank <= kPermuteCodegenMaxDims, "Permute codegen supports rank up to {}, got {}", kPermuteCodegenMaxDims, rank);
    return ttsl::SmallVector<uint32_t>(params.dims.begin(), params.dims.begin() + rank);
}

}  // namespace

PermuteCodegenDeviceOperation::program_factory_t PermuteCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return PermuteCodegenProgramFactory{};
}

// The row-invariant CB plan is read from the live L1 frontier, so it is hashed; a hit on a stale
// depth would overrun the CB region. The hash runs before validation, so the plan is only read once
// the input is a valid device tensor and dims is a permutation of at least two axes.
ttsl::hash::hash_t PermuteCodegenDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input = tensor_args.input;
    const uint32_t rank = input.logical_shape().rank();
    const auto dims = active_dims(operation_attributes, rank);
    PermuteRmCbPlan plan;
    if (input.storage_type() == ttnn::StorageType::DEVICE && input.buffer() != nullptr &&
        rank >= kPermuteHeightAxisFromEnd &&
        ttnn::operations::data_movement::permute_codegen::is_valid_permutation(dims, rank) &&
        dims[rank - kPermuteWidthAxisFromEnd] == rank - kPermuteWidthAxisFromEnd) {
        plan = permute_rm_cb_plan(input, compute_output_specs(operation_attributes, tensor_args))
                   .value_or(PermuteRmCbPlan{});
    }
    return ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<PermuteCodegenDeviceOperation>,
        operation_attributes,
        tensor_args,
        plan.batch,
        plan.depth);
}

void PermuteCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input = tensor_args.input;
    TT_FATAL(input.storage_type() == ttnn::StorageType::DEVICE, "Operands to permute need to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    const uint32_t rank = input.logical_shape().rank();

    namespace codegen = ttnn::operations::data_movement::permute_codegen;
    const auto dims = active_dims(operation_attributes, rank);
    TT_FATAL(
        std::all_of(dims.cbegin(), dims.cend(), [rank](uint32_t d) { return d < rank; }),
        "Permute dims must be within the input rank {}",
        rank);
    TT_FATAL(codegen::is_valid_permutation(dims, rank), "Permute dims must be a permutation of the input axes");

    if (tensor_args.optional_output_tensor.has_value()) {
        const Tensor& output = *tensor_args.optional_output_tensor;
        TT_FATAL(output.storage_type() == ttnn::StorageType::DEVICE, "Optional output tensor must be on device");
        TT_FATAL(output.buffer() != nullptr, "Optional output tensor must be allocated in a buffer on device");
        TT_FATAL(output.device() == input.device(), "Optional output tensor must be on the same device as the input");
        TT_FATAL(
            codegen::output_spec_matches(
                codegen::permute_output_spec(input, dims, operation_attributes.output_mem_config),
                output.tensor_spec()),
            "Optional output tensor does not match the permuted input layout, dtype, tile and memory config");
    }

    TT_FATAL(
        codegen::supported_by_codegen(input, dims, operation_attributes.output_mem_config),
        "Input is not supported by PermuteCodegen");
}

PermuteCodegenDeviceOperation::spec_return_value_t PermuteCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }
    const auto& input = tensor_args.input;
    return ttnn::operations::data_movement::permute_codegen::permute_output_spec(
        input, active_dims(operation_attributes, input.logical_shape().rank()), operation_attributes.output_mem_config);
}

PermuteCodegenDeviceOperation::tensor_return_value_t PermuteCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> PermuteCodegenDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    int ideal_dev_clock_cycles = operations::data_movement::common_tm_bw_model(input_tensor, output_tensor);
    return {{input_tensor}, output_tensor, ideal_dev_clock_cycles};
}

PermuteCodegenDeviceOperation::tensor_return_value_t permute_codegen(
    const Tensor& input, const PermuteCodegenParams& params, std::optional<Tensor> optional_output_tensor) {
    using OperationType = PermuteCodegenDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params,
        OperationType::tensor_args_t{.input = input, .optional_output_tensor = std::move(optional_output_tensor)});
}

}  // namespace ttnn::prim
