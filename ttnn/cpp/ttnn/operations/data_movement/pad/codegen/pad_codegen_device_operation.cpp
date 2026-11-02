// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/pad/codegen/pad_codegen_device_operation.hpp"

#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/pad/codegen/pad_codegen_supported.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt_stl/reflection.hpp>

namespace ttnn::prim {

PadCodegenDeviceOperation::program_factory_t PadCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return PadCodegenProgramFactory{};
}

namespace {

void validate_tensors(const PadCodegenParams& operation_attributes, const PadCodegenInputs& tensor_args) {
    const Tensor& input = tensor_args.input;
    TT_FATAL(input.storage_type() == ttnn::StorageType::DEVICE, "Operands to pad need to be on device!");
    TT_FATAL(input.is_allocated(), "Operands need to be allocated in buffers on device!");
    const auto& optional_output = tensor_args.optional_output_tensor;
    if (optional_output.has_value()) {
        TT_FATAL(optional_output->storage_type() == ttnn::StorageType::DEVICE, "Output of pad needs to be on device!");
        TT_FATAL(optional_output->is_allocated(), "Output of pad needs to be allocated in a buffer on device!");
        TT_FATAL(optional_output->device() == input.device(), "Input and output of pad must be on the same device");
        TT_FATAL(
            optional_output->tensor_spec() ==
                ttnn::operations::data_movement::pad_codegen::codegen_output_spec(input, operation_attributes),
            "Output tensor spec does not match the spec pad builds for this input");
    }
}

}  // namespace

void PadCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
    const Tensor& input = tensor_args.input;
    const auto& optional_output = tensor_args.optional_output_tensor;
    TT_FATAL(
        !ttnn::operations::data_movement::pad_codegen::composes_via_bf16(input, operation_attributes),
        "Nonzero pad of a block-float tensor must be padded through a bfloat16 copy");
    TT_FATAL(
        ttnn::operations::data_movement::pad_codegen::supported_by_codegen(
            input, operation_attributes, optional_output),
        "Input is not supported by PadCodegen");
}

// Everything the support gate reads is part of the program cache key, so a hit only re-checks the tensors.
void PadCodegenDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
}

// The CB batching follows the L1 free right now (plan_batches_live runs on every dispatch), so it is part of the key:
// an occupancy change that moves the plan is a miss that rebuilds, not a hit that replays CBs sized for L1 that is no
// longer free. Because the hash depends on live L1, a trace captured under one L1 state replays that state's plan.
ttsl::hash::hash_t PadCodegenDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto batches =
        ttnn::operations::data_movement::pad_codegen::plan_batches_live(tensor_args.input, operation_attributes);
    const uint32_t read_batch = batches.has_value() ? batches->read_batch : 0;
    const uint32_t write_batch = batches.has_value() ? batches->write_batch : 0;
    return ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<PadCodegenDeviceOperation>, operation_attributes, tensor_args, read_batch, write_batch);
}

PadCodegenDeviceOperation::spec_return_value_t PadCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }
    return ttnn::operations::data_movement::pad_codegen::codegen_output_spec(tensor_args.input, operation_attributes);
}

PadCodegenDeviceOperation::tensor_return_value_t PadCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> PadCodegenDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    int ideal_dev_clock_cycles = operations::data_movement::common_tm_bw_model(input_tensor, output_tensor);
    return {{input_tensor}, output_tensor, ideal_dev_clock_cycles};
}

PadCodegenDeviceOperation::tensor_return_value_t pad_codegen(
    const Tensor& input, const PadCodegenParams& params, std::optional<Tensor> optional_output_tensor) {
    using OperationType = PadCodegenDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params,
        OperationType::tensor_args_t{.input = input, .optional_output_tensor = std::move(optional_output_tensor)});
}

}  // namespace ttnn::prim
