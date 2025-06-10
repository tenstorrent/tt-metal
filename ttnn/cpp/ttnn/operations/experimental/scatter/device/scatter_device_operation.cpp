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
    const uint32_t src_rank{src_shape.rank()};

    TT_FATAL(
        args.dim < static_cast<int32_t>(input_rank) && -static_cast<int32_t>(input_rank) <= args.dim,
        "dim must follow the condition -input_rank <= dim < input_rank (dim: {}, rank: {}).",
        args.dim,
        static_cast<int32_t>(input_rank));

    if (tensor_args.opt_output.has_value()) {
        const auto& output_tensor{tensor_args.opt_output.value()};
        const auto& output_shape{output_tensor.logical_shape()};
        const auto& output_rank{output_shape.rank()};
        const auto& output_dtype{output_tensor.dtype()};

        TT_FATAL(
            input_shape == output_shape,
            "The shapes of input and output tensors must be equal (input_shape: {}, output_shape: {}).",
            input_shape,
            output_shape);

        TT_FATAL(
            input_dtype == output_dtype,
            "input_dtype and output_dtype must be the same (input_dtype: {}, output_dtype: {})",
            magic_enum::enum_name(input_dtype),
            magic_enum::enum_name(output_dtype));

        TT_FATAL(
            static_cast<int32_t>(args.dim) >= -static_cast<int32_t>(output_rank),
            "dim cannot be lower than output shape's negative rank (dim: {}, rank: {}).",
            args.dim,
            -static_cast<int32_t>(output_rank));
        TT_FATAL(
            static_cast<int32_t>(args.dim) < static_cast<int32_t>(output_rank),
            "dim must be lower than output shape's positive rank (dim: {}, rank: {}).",
            args.dim,
            static_cast<int32_t>(output_rank));

        TT_FATAL(
            output_tensor.layout() == Layout::TILE,
            "Output tensor doesn't have a tile layout - only tile layout is supported.");
        TT_FATAL(output_tensor.buffer() != nullptr, "Output tensor's buffer is null.");
        TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Output tensor must be allocated on a device.");
        TT_FATAL(!output_tensor.is_sharded(), "Sharded tensors are not supported - output_tensor is sharded.");
    }

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

    const int32_t dim{(args.dim < 0) ? (args.dim + input_tensor.padded_shape().rank()) : args.dim};

    for (uint32_t probe_dim = 0; probe_dim < input_shape.rank(); ++probe_dim) {
        if (probe_dim != dim) {
            TT_FATAL(
                index_shape[probe_dim] == input_shape[probe_dim],
                "Index tensor has other dimension {}'s length than input shape's (index dimension: {}, "
                "input_dimension: "
                "{}).",
                probe_dim,
                index_shape[probe_dim],
                input_shape[probe_dim]);
        }
    }

    TT_FATAL(!input_tensor.is_sharded(), "Sharded tensors are not supported - input_tensor is sharded.");
    TT_FATAL(!index_tensor.is_sharded(), "Sharded tensors are not supported - index_tensor is sharded.");
    TT_FATAL(!src_tensor.is_sharded(), "Sharded tensors are not supported - src_tensor is sharded.");

    TT_FATAL(
        input_tensor.layout() == Layout::TILE,
        "Input tensor doesn't have a tile layout - only tile layout is supported.");
    TT_FATAL(
        index_tensor.layout() == Layout::TILE,
        "Index tensor doesn't have a tile layout - only tile layout is supported.");
    TT_FATAL(
        src_tensor.layout() == Layout::TILE, "Src tensor doesn't have a tile layout - only tile layout is supported.");

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
    if (tensor_args.opt_output.has_value()) {
        return tensor_args.opt_output.value().tensor_spec();
    }

    return TensorSpec{
        tensor_args.input_tensor.logical_shape(),
        TensorLayout{tensor_args.input_tensor.dtype(), PageConfig{Layout::TILE}, args.output_memory_config}};
}

ScatterDeviceOperation::tensor_return_value_t ScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.opt_output.has_value()) {
        return *tensor_args.opt_output;
    }

    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device());
}

ScatterDeviceOperation::invocation_result_t ScatterDeviceOperation::invoke(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const MemoryConfig& output_memory_config,
    const std::optional<ScatterReductionType>& opt_reduction,
    std::optional<Tensor>& opt_output,
    const QueueId& queue_id) {
    return {
        operation_attributes_t{dim, output_memory_config, opt_reduction},
        tensor_args_t{input_tensor, index_tensor, source_tensor, opt_output}};
}

}  // namespace ttnn::operations::experimental::scatter
