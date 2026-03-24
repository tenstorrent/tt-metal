// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "dit_minimal_binary_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

void DitMinimalRmBinaryDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input_a;
    const auto& b = tensor_args.input_b;

    TT_FATAL(
        a.storage_type() == StorageType::DEVICE && b.storage_type() == StorageType::DEVICE,
        "dit_minimal_binary: both inputs must be on device");

    TT_FATAL(
        a.buffer() != nullptr && b.buffer() != nullptr,
        "dit_minimal_binary: both inputs must be allocated in buffers on device");

    TT_FATAL(
        a.layout() == Layout::ROW_MAJOR && b.layout() == Layout::ROW_MAJOR,
        "dit_minimal_binary: both inputs must be ROW_MAJOR layout");

    TT_FATAL(
        a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
            b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "dit_minimal_binary: both inputs must be INTERLEAVED (no sharded support)");

    TT_FATAL(
        a.memory_config().buffer_type() == BufferType::DRAM && b.memory_config().buffer_type() == BufferType::DRAM,
        "dit_minimal_binary: both inputs must be in DRAM");

    TT_FATAL(
        a.dtype() == DataType::BFLOAT16 || a.dtype() == DataType::FLOAT32,
        "dit_minimal_binary: only bfloat16 and float32 are supported, got {}",
        static_cast<int>(a.dtype()));

    TT_FATAL(
        a.dtype() == b.dtype(),
        "dit_minimal_binary: dtype mismatch: {} vs {}",
        static_cast<int>(a.dtype()),
        static_cast<int>(b.dtype()));

    TT_FATAL(a.padded_shape() == b.padded_shape(), "dit_minimal_binary: shape mismatch");

    TT_FATAL(a.physical_volume() > 0, "dit_minimal_binary: tensor has 0 elements");
}

DitMinimalRmBinaryDeviceOperation::spec_return_value_t DitMinimalRmBinaryDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return TensorSpec(
        tensor_args.input_a.logical_shape(),
        TensorLayout(args.output_dtype, Layout::ROW_MAJOR, args.output_memory_config));
}

DitMinimalRmBinaryDeviceOperation::tensor_return_value_t DitMinimalRmBinaryDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_a.device());
}

tt::stl::hash::hash_t DitMinimalRmBinaryDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<DitMinimalRmBinaryDeviceOperation>(
        static_cast<uint8_t>(args.op_type),
        tensor_args.input_a.dtype(),
        tensor_args.input_a.memory_config(),
        tensor_args.input_a.padded_shape()[-1]);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor dit_minimal_binary(
    const Tensor& input_a,
    const Tensor& input_b,
    ttnn::experimental::prim::BinaryOpType op_type,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& output_memory_config) {
    using OperationType = ttnn::experimental::prim::DitMinimalRmBinaryDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .op_type = op_type,
        .output_dtype = output_dtype,
        .output_memory_config = output_memory_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .input_a = input_a,
        .input_b = input_b,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
