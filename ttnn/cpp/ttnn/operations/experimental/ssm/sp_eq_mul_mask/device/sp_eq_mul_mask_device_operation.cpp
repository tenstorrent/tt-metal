// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sp_eq_mul_mask_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

void SpEqMulMaskDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;

    TT_FATAL(a.layout() == Layout::TILE, "Input A must be tilized");
    TT_FATAL(b.layout() == Layout::TILE, "Input B must be tilized");
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Input A must be on device");
    TT_FATAL(b.storage_type() == StorageType::DEVICE, "Input B must be on device");
    TT_FATAL(a.buffer() != nullptr && b.buffer() != nullptr, "Both inputs must be allocated on device");
    TT_FATAL(
        a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
            b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only interleaved inputs supported");
    TT_FATAL(
        a.dtype() == DataType::BFLOAT16 && b.dtype() == DataType::BFLOAT16,
        "Only bfloat16 inputs supported");
    TT_FATAL(a.padded_shape() == b.padded_shape(), "Inputs must have identical padded shapes");
    const auto& shape = a.padded_shape();
    TT_FATAL(shape[-1] % TILE_WIDTH == 0 && shape[-2] % TILE_HEIGHT == 0,
             "Input shape must be tile-aligned on last two dims");
    TT_FATAL(
        args.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only interleaved output supported");
}

SpEqMulMaskDeviceOperation::spec_return_value_t SpEqMulMaskDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return TensorSpec(
        tensor_args.a.padded_shape(),
        TensorLayout(args.dtype, PageConfig(Layout::TILE), args.memory_config));
}

SpEqMulMaskDeviceOperation::tensor_return_value_t SpEqMulMaskDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.a.device());
}

ttsl::hash::hash_t SpEqMulMaskDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.a;
    return operation::hash_operation<SpEqMulMaskDeviceOperation>(
        args, a.dtype(), a.memory_config(), a.padded_shape().volume());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor sp_eq_mul_mask(
    const Tensor& a,
    const Tensor& b,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::DataType> dtype) {
    using OpT = ttnn::experimental::prim::SpEqMulMaskDeviceOperation;
    auto attrs = OpT::operation_attributes_t{
        .memory_config = memory_config.value_or(a.memory_config()),
        .dtype = dtype.value_or(a.dtype()),
    };
    auto inputs = OpT::tensor_args_t{.a = a, .b = b};
    return ttnn::device_operation::launch<OpT>(attrs, inputs);
}

}  // namespace ttnn::prim
