// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "iterative_topk_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::iterative_topk {

void IterativeTopkDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    TT_FATAL(input.storage_type() == tt::tt_metal::StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input.buffer() != nullptr, "Input tensor must be allocated");
    TT_FATAL(input.dtype() == tt::tt_metal::DataType::FLOAT32, "Input tensor must be FLOAT32");
    TT_FATAL(
        input.layout() == tt::tt_metal::Layout::ROW_MAJOR || input.layout() == tt::tt_metal::Layout::TILE,
        "Input tensor must be ROW_MAJOR or TILE layout");

    const auto memory_layout = input.memory_config().memory_layout();

    if (input.layout() == tt::tt_metal::Layout::TILE) {
        TT_FATAL(
            memory_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED, "TILE layout input must be interleaved");
        uint32_t width = input.logical_shape()[-1];
        uint32_t num_rows = input.logical_shape().volume() / width;
        TT_FATAL(width % 32 == 0, "TILE layout input width ({}) must be divisible by 32", width);
        TT_FATAL(num_rows % 32 == 0, "TILE layout input num_rows ({}) must be divisible by 32", num_rows);
    } else {
        TT_FATAL(
            memory_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED ||
                memory_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            "Input tensor must be interleaved or height-sharded");

        if (memory_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
            TT_FATAL(input.shard_spec().has_value(), "Height-sharded input must have a shard spec");
            const auto& shard_spec = input.shard_spec().value();
            uint32_t width = input.logical_shape()[-1];
            TT_FATAL(
                shard_spec.shape[1] == width,
                "Height-sharded input shard width ({}) must equal tensor width ({})",
                shard_spec.shape[1],
                width);
            TT_FATAL(shard_spec.shape[0] > 0, "Shard height must be > 0");
        }
    }

    TT_FATAL(attributes.k > 0, "k must be > 0");
    TT_FATAL(
        attributes.k <= input.logical_shape()[-1],
        "k ({}) must be <= last dimension size ({})",
        attributes.k,
        input.logical_shape()[-1]);
    TT_FATAL(input.logical_shape().rank() >= 2, "Input tensor must be at least 2D");
}

IterativeTopkDeviceOperation::program_factory_t IterativeTopkDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& tensor_args) {
    if (tensor_args.input.layout() == tt::tt_metal::Layout::TILE) {
        return TiledProgramFactory{};
    }
    if (tensor_args.input.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
        return ShardedProgramFactory{};
    }
    return ProgramFactory{};
}

IterativeTopkDeviceOperation::spec_return_value_t IterativeTopkDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    auto shape = input.logical_shape();
    auto output_shape = shape;
    output_shape[-1] = attributes.k;

    auto output_mem_config = attributes.output_mem_config;

    if (input.layout() == tt::tt_metal::Layout::TILE) {
        return std::array<TensorSpec, 2>{
            TensorSpec(
                output_shape,
                tt::tt_metal::TensorLayout(
                    tt::tt_metal::DataType::FLOAT32,
                    tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                    output_mem_config)),
            TensorSpec(
                output_shape,
                tt::tt_metal::TensorLayout(
                    tt::tt_metal::DataType::UINT16,
                    tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                    output_mem_config))};
    }

    if (input.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
        const auto& input_shard_spec = input.shard_spec().value();
        auto output_shard_spec = input_shard_spec;
        output_shard_spec.shape[1] = attributes.k;
        output_mem_config = tt::tt_metal::MemoryConfig{
            tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED, tt::tt_metal::BufferType::L1, output_shard_spec};
    }

    return std::array<TensorSpec, 2>{
        TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                output_mem_config)),
        TensorSpec(
            output_shape,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::UINT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
                output_mem_config))};
}

IterativeTopkDeviceOperation::tensor_return_value_t IterativeTopkDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(attributes, tensor_args);
    return std::array<Tensor, 2>{
        create_device_tensor(specs[0], tensor_args.input.device()),
        create_device_tensor(specs[1], tensor_args.input.device())};
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::iterative_topk

namespace ttnn::prim {

ttnn::operations::experimental::deepseek_prefill::iterative_topk::IterativeTopkDeviceOperation::tensor_return_value_t
iterative_topk(const Tensor& input, uint32_t k, const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::iterative_topk::IterativeTopkDeviceOperation;

    auto operation_attributes =
        OperationType::operation_attributes_t{k, output_mem_config.value_or(input.memory_config())};
    auto tensor_args = OperationType::tensor_args_t{input};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
