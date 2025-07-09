// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_device_operation.hpp"
#include "scatter_program_factory.hpp"

#include <magic_enum/magic_enum.hpp>

namespace ttnn::operations::experimental::scatter {

ScatterDeviceOperation::program_factory_t ScatterDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ScatterProgramFactory{};
}

void ScatterDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void ScatterDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor{tensor_args.input_tensor};
    const auto& index_tensor{tensor_args.index_tensor};
    const auto& src_tensor{tensor_args.src_tensor};
    const auto& input_dtype{input_tensor.dtype()};
    const auto& index_dtype{index_tensor.dtype()};
    const auto& src_dtype{src_tensor.dtype()};
    const auto& input_shape{input_tensor.logical_shape()};
    const auto& index_shape{index_tensor.logical_shape()};
    const auto& src_shape{src_tensor.logical_shape()};
    const uint32_t input_rank{input_shape.rank()};
    const uint32_t index_rank{index_shape.rank()};

    TT_FATAL(
        index_shape == src_shape,
        "index_shape must be the same as src_shape (index_shape: {}, src_shape: {}).",
        index_shape,
        src_shape);

    TT_FATAL(
        input_rank == index_rank,
        "input_rank must equal index_rank (input_rank: {}, index_rank: {}).",
        input_rank,
        index_rank);

    TT_FATAL(
        input_dtype == src_dtype,
        "input_dtype differs from src_dtype (input_dtype: {}, src_dtype: {}).",
        magic_enum::enum_name(input_dtype),
        magic_enum::enum_name(src_dtype));

    TT_FATAL(
        index_dtype == DataType::INT32 || index_dtype == DataType::UINT8 || index_dtype == DataType::UINT16 ||
            index_dtype == DataType::UINT32,
        "index_dtype is not integer, it is {}.",
        magic_enum::enum_name(index_dtype));

    TT_FATAL(!input_tensor.is_sharded(), "Sharded tensors are not supported - input_tensor is sharded.");
    TT_FATAL(!index_tensor.is_sharded(), "Sharded tensors are not supported - index_tensor is sharded.");
    TT_FATAL(!src_tensor.is_sharded(), "Sharded tensors are not supported - src_tensor is sharded.");

    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor's buffer is null.");
    TT_FATAL(index_tensor.buffer() != nullptr, "Index tensor's buffer is null.");
    TT_FATAL(src_tensor.buffer() != nullptr, "Src tensor's buffer is null.");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be allocated on a device.");
    TT_FATAL(index_tensor.storage_type() == StorageType::DEVICE, "Index tensor must be allocated on a device.");
    TT_FATAL(src_tensor.storage_type() == StorageType::DEVICE, "Src tensor must be allocated on a device.");
}

ScatterDeviceOperation::spec_return_value_t ScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;
    return TensorSpec{
        tensor_args.input_tensor.get_logical_shape(),
        TensorLayout{tensor_args.input_tensor.get_dtype(), PageConfig{Layout::ROW_MAJOR}, args.output_memory_config}};
}

ScatterDeviceOperation::tensor_return_value_t ScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

ScatterDeviceOperation::invocation_result_t ScatterDeviceOperation::invoke(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const MemoryConfig& output_memory_config,
    const std::optional<ScatterReductionType>& opt_reduction,
    const QueueId& queue_id) {
    return {
        operation_attributes_t{dim, output_memory_config, opt_reduction},
        tensor_args_t{input_tensor, index_tensor, source_tensor}};
}

operation::Hash ScatterDeviceOperation::compute_program_hash(
    const operation_attributes_t& op_args, const tensor_args_t& tensor_args) {
    return operation::hash_operation<ScatterDeviceOperation>(
        select_program_factory(op_args, tensor_args).index(),
        op_args.dim,
        op_args.opt_reduction,
        op_args.output_memory_config,
        tensor_args.input_tensor.logical_shape(),
        tensor_args.index_tensor.logical_shape(),
        tensor_args.src_tensor.logical_shape(),
        tensor_args.input_tensor.dtype(),
        tensor_args.index_tensor.dtype(),
        tensor_args.src_tensor.dtype(),
        tensor_args.input_tensor.memory_config(),
        tensor_args.index_tensor.memory_config(),
        tensor_args.src_tensor.memory_config(),
        tensor_args.input_tensor.layout(),
        tensor_args.index_tensor.layout(),
        tensor_args.src_tensor.layout());
}

}  // namespace ttnn::operations::experimental::scatter
