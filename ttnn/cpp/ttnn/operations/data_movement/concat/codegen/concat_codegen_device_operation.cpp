// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_supported.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt_stl/reflection.hpp>

namespace ttnn::prim {

ConcatCodegenDeviceOperation::program_factory_t ConcatCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return ConcatCodegenProgramFactory{};
}

namespace {

// The structural checks the native concat applies, so a forced call fails the same way on malformed input.
void validate_tensors(const ConcatCodegenParams& operation_attributes, const ConcatCodegenInputs& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensors;
    TT_FATAL(!input_tensors.empty(), "need 1 or more tensors");

    const auto& first_input = input_tensors[0];
    auto shape_first = first_input.logical_shape();
    TT_FATAL(
        operation_attributes.dim < shape_first.rank(),
        "ConcatCodegenDeviceOperation dim specified is larger than input tensor rank.");
    shape_first[operation_attributes.dim] = 0;

    for (const auto& in_ref : input_tensors) {
        TT_FATAL(in_ref.storage_type() == ttnn::StorageType::DEVICE, "Operand to concat needs to be on device.");
        TT_FATAL(in_ref.buffer() != nullptr, "Operand to concat needs to be allocated in a buffer on device.");
        TT_FATAL(in_ref.device() == first_input.device(), "Operands to concat need to be on the same device.");
        TT_FATAL(in_ref.layout() == first_input.layout(), "All Tensors should have same layouts.");
        TT_FATAL(in_ref.dtype() == first_input.dtype(), "All Tensors should have same dtypes.");
        auto curr_shape = in_ref.logical_shape();
        TT_FATAL(curr_shape.rank() == shape_first.rank(), "Input tensor ranks must be equal");
        curr_shape[operation_attributes.dim] = 0;
        TT_FATAL(curr_shape == shape_first, "concat tensors differ in shape across non-concat dimensions.");
        TT_FATAL(
            in_ref.is_sharded() == first_input.is_sharded(), "All tensors must be sharded or all must be interleaved");
    }

    const auto& optional_output = tensor_args.optional_output_tensor;
    if (optional_output.has_value()) {
        TT_FATAL(
            optional_output->storage_type() == ttnn::StorageType::DEVICE, "Output of concat needs to be on device!");
        TT_FATAL(optional_output->buffer() != nullptr, "Output of concat needs to be allocated in a buffer on device!");
        TT_FATAL(
            optional_output->device() == first_input.device(),
            "Inputs and output of concat must be on the same device");
        TT_FATAL(
            optional_output->tensor_spec() == ttnn::operations::data_movement::concat_codegen::codegen_output_spec(
                                                  input_tensors, operation_attributes),
            "Output tensor spec does not match the spec concat builds for these inputs");
    }
}

}  // namespace

void ConcatCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
    TT_FATAL(
        ttnn::operations::data_movement::concat_codegen::supported_by_codegen(
            tensor_args.input_tensors, operation_attributes, tensor_args.optional_output_tensor),
        "ConcatCodegen does not support this input");
}

// Everything the support gate reads is part of the program cache key, so a hit only re-checks the tensors.
void ConcatCodegenDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
}

// The CB batching follows the L1 free right now (plan_batches_live runs on every dispatch), so it is part of the key:
// an occupancy change that moves the plan is a miss that rebuilds, not a hit that replays CBs sized for L1 that is no
// longer free. Because the hash depends on live L1, a trace captured under one L1 state replays that state's plan.
ttsl::hash::hash_t ConcatCodegenDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensors = tensor_args.input_tensors;
    uint32_t read_batch = 0;
    uint32_t write_batch = 0;
    if (!input_tensors.empty() && input_tensors.front().storage_type() == ttnn::StorageType::DEVICE &&
        input_tensors.front().buffer() != nullptr) {
        const auto batches =
            ttnn::operations::data_movement::concat_codegen::plan_batches_live(input_tensors, operation_attributes);
        if (batches.has_value()) {
            read_batch = batches->read_batch;
            write_batch = batches->write_batch;
        }
    }

    // Each input's position of the first input backed by the same buffer. The program is cached with one buffer
    // address slot per input, so concat([x, x]) and concat([a, b]) of identical specs must not share an entry.
    std::vector<uint32_t> input_alias_signature(input_tensors.size());
    for (uint32_t i = 0; i < input_tensors.size(); ++i) {
        input_alias_signature[i] = i;
        for (uint32_t j = 0; j < i; ++j) {
            if (&input_tensors[j].mesh_tensor() == &input_tensors[i].mesh_tensor()) {
                input_alias_signature[i] = j;
                break;
            }
        }
    }

    return ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<ConcatCodegenDeviceOperation>,
        operation_attributes,
        tensor_args,
        read_batch,
        write_batch,
        input_alias_signature);
}

ConcatCodegenDeviceOperation::spec_return_value_t ConcatCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }
    TT_FATAL(!tensor_args.input_tensors.empty(), "concat needs at least one input tensor");
    return ttnn::operations::data_movement::concat_codegen::codegen_output_spec(
        tensor_args.input_tensors, operation_attributes);
}

ConcatCodegenDeviceOperation::tensor_return_value_t ConcatCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensors.at(0).device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> ConcatCodegenDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensors.at(0);
    int ideal_dev_clock_cycles = operations::data_movement::common_tm_bw_model(
        input_tensor, output_tensor, false, 0, false, false, false, /*concat_op=*/true);
    return {tensor_args.input_tensors, output_tensor, ideal_dev_clock_cycles};
}

ConcatCodegenDeviceOperation::tensor_return_value_t concat_codegen(
    const std::vector<Tensor>& input_tensors,
    const ConcatCodegenParams& params,
    std::optional<Tensor> optional_output_tensor) {
    using OperationType = ConcatCodegenDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params,
        OperationType::tensor_args_t{
            .input_tensors = input_tensors, .optional_output_tensor = std::move(optional_output_tensor)});
}

}  // namespace ttnn::prim
