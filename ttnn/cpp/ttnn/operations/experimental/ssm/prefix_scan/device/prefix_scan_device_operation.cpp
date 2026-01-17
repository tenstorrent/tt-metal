// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prefix_scan_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::ssm::prefix_scan {

PrefixScanDeviceOperation::program_factory_t PrefixScanDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return program::PrefixScanProgramFactory{};
}

void PrefixScanDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void PrefixScanDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& a = tensor_args.a;
    const auto& bx = tensor_args.bx;
    const auto& h_prev = tensor_args.h_prev;

    TT_FATAL(a.dtype() == bx.dtype(), "Expected input tensors to have the same data type");
    TT_FATAL(a.layout() == Layout::TILE && bx.layout() == Layout::TILE, "Expected input tensors to be tile layout");
    TT_FATAL(a.padded_shape() == bx.padded_shape(), "Expected input tensors to have the same shape");

    const auto& shape = a.padded_shape();
    TT_FATAL(shape.rank() == 4, "Expected input tensors to be rank 4");
    TT_FATAL(shape[0] == 1 && shape[1] == 1, "Dimension 0 and 1 should be size 1");
    TT_FATAL(
        shape[2] >= tt::constants::TILE_HEIGHT && shape[2] % tt::constants::TILE_HEIGHT == 0,
        "Sequence length should be a multiple of 32");

    TT_FATAL(h_prev.dtype() == DataType::BFLOAT16, "Expected initial hidden state to be bfloat16");
    TT_FATAL(h_prev.layout() == Layout::ROW_MAJOR, "Expected initial hidden state to be row-major");

    TT_FATAL(a.is_sharded() && bx.is_sharded() && h_prev.is_sharded(), "Expected input tensors to be sharded");
    TT_FATAL(
        a.shard_spec().has_value() && bx.shard_spec().has_value() && h_prev.shard_spec().has_value(),
        "Expected input tensors to be sharded");
    TT_FATAL(
        a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
        "Expected A tensor to be row major orientation");
    TT_FATAL(
        bx.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
        "Expected Bx tensor to be row major orientation");
    TT_FATAL(
        h_prev.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
        "Expected h tensor to be row major orientation");
}

TensorSpec PrefixScanDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.a;
    return TensorSpec(
        a.logical_shape(),
        TensorLayout::fromPaddedShape(
            args.dtype, PageConfig(Layout::TILE), args.memory_config, a.logical_shape(), a.padded_shape()));
}

Tensor PrefixScanDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.a.device());
}

tt::stl::hash::hash_t PrefixScanDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.a;
    const auto& a_shape = a.padded_shape();

    auto program_factory = select_program_factory(args, tensor_args);
    operation::Hash hash = operation::hash_operation<PrefixScanDeviceOperation>(
        args.math_fidelity, program_factory.index(), a.dtype(), a.memory_config(), a_shape.volume());

    return hash;
}

}  // namespace ttnn::operations::experimental::ssm::prefix_scan

namespace ttnn::prim {

ttnn::operations::experimental::ssm::prefix_scan::PrefixScanDeviceOperation::tensor_return_value_t prefix_scan(
    const Tensor& a,
    const Tensor& bx,
    const Tensor& h_prev,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> dtype,
    std::optional<MathFidelity> math_fidelity) {
    using OperationType = ttnn::operations::experimental::ssm::prefix_scan::PrefixScanDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .memory_config = memory_config.value_or(a.memory_config()),
        .dtype = dtype.value_or(a.dtype()),
        .math_fidelity = math_fidelity.value_or(MathFidelity::HiFi4),
    };
    auto tensor_args = OperationType::tensor_args_t{.a = a, .bx = bx, .h_prev = h_prev};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
