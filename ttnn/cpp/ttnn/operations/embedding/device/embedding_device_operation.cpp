// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/embedding/device/embedding_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

#define RISC_CORES_PER_TENSIX 2

namespace ttnn::prim {

EmbeddingsDeviceOperation::program_factory_t EmbeddingsDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input_tensor_arg.layout() == ttnn::TILE_LAYOUT) {
        return EmbeddingsTilizedIndicesProgramFactory{};
    }
    if (operation_attributes.tilized) {
        return EmbeddingsFusedProgramFactory{};
    }
    return EmbeddingsRMProgramFactory{};
}

void EmbeddingsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void EmbeddingsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input_tensor_arg;
    const auto& weights = tensor_args.weight_arg;

    TT_FATAL(
        weights.layout() == Layout::ROW_MAJOR, "Weights tensor layout must be ROW_MAJOR but got {}", weights.layout());
    TT_FATAL(a.dtype() == DataType::UINT32 or a.dtype() == DataType::BFLOAT16, "Input must be UINT32 or BFLOAT16");
    TT_FATAL(weights.dtype() == DataType::BFLOAT16, "Weights tensor must have BFLOAT16 dtype");
    TT_FATAL(
        a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Embedding does not currently support sharded inputs");
    TT_FATAL(
        weights.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Embedding does not currently support sharded weights");
    TT_FATAL(
        weights.padded_shape()[0] == 1 && weights.padded_shape()[1] == 1,
        "First two dimensions for the weights must be 1 but got {} and {}",
        weights.padded_shape()[0],
        weights.padded_shape()[1]);
    if (operation_attributes.tilized) {
        TT_FATAL(
            a.padded_shape()[-1] % TILE_HEIGHT == 0,
            "Input tensor width {} must be a multiple of tile height {} to have the output tensor tilized",
            a.padded_shape()[-1],
            TILE_HEIGHT);
        TT_FATAL(
            weights.padded_shape()[-1] % TILE_WIDTH == 0,
            "Number of columns in table {} must be factor of tile width {}",
            weights.padded_shape()[-1],
            TILE_WIDTH);
        if (is_sharded(operation_attributes.output_mem_config.memory_layout())) {
            const auto& shard_spec = operation_attributes.output_mem_config.shard_spec();
            TT_FATAL(shard_spec.has_value(), "Sharded memory config must have a shard spec");
            TT_FATAL(
                shard_spec->shape[0] % TILE_HEIGHT == 0,
                "Shard height {} must be a multiple of tile height {} to have the output tensor tilized",
                shard_spec->shape[0],
                TILE_HEIGHT);
            TT_FATAL(
                shard_spec->shape[1] % TILE_WIDTH == 0,
                "Shard width {} must be a multiple of tile width {} to have the output tensor tilized",
                shard_spec->shape[1],
                TILE_WIDTH);
            TT_FATAL(
                a.physical_volume() % shard_spec->shape[0] == 0,
                "Input tensor volume {} must be a multiple of shard height {}",
                a.physical_volume(),
                shard_spec->shape[0]);
            TT_FATAL(
                weights.padded_shape()[-1] % shard_spec->shape[1] == 0,
                "Number of columns in table {} must be factor of shard width {}",
                weights.padded_shape()[-1],
                shard_spec->shape[1]);
        }
    } else {
        if (is_sharded(operation_attributes.output_mem_config.memory_layout())) {
            TT_FATAL(
                operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                "Embedding only supports height sharded Row Major outputs");
        }
    }
    if (a.layout() == Layout::ROW_MAJOR) {
        TT_FATAL(a.padded_shape()[1] == 1 && a.padded_shape()[2] == 1, "Only dim 0 && 3 for the input can be non 1");
    }
    switch (operation_attributes.embeddings_type) {
        case EmbeddingsType::PADDED:
            TT_FATAL(
                operation_attributes.pad_token.has_value(),
                "Pad token must be specified when PADDED Embeddings Type is specified");
            break;
        case EmbeddingsType::BINARY:
            TT_FATAL(
                weights.padded_shape()[-2] == 2, "Weight tensor must have 2 embeddings for BINARY Embeddings Type");
            break;
        default:
            TT_FATAL(
                !operation_attributes.pad_token.has_value(),
                "Pad token must not be specified when PADDED Embeddings Type is not specified");
    }
}

TensorSpec EmbeddingsDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor_arg;
    const auto& weight_tensor = tensor_args.weight_arg;
    auto num_output_embeddings = input_tensor.logical_shape()[-1];
    auto batch_num = input_tensor.logical_shape()[0];
    auto num_embedding_dims = weight_tensor.logical_shape()[-1];

    ttnn::Shape output_shape({batch_num, 1, num_output_embeddings, num_embedding_dims});
    auto output_layout =
        (operation_attributes.tilized && input_tensor.layout() != Layout::TILE) ? Layout::TILE : Layout::ROW_MAJOR;
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }
    return TensorSpec(
        output_shape,
        TensorLayout(weight_tensor.dtype(), PageConfig(output_layout), operation_attributes.output_mem_config));
}

Tensor EmbeddingsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return *tensor_args.optional_output_tensor;
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor_arg.device());
}

Tensor embedding(
    const Tensor& input_tensor_arg,
    const Tensor& weight_arg,
    bool tilized,
    EmbeddingsType embeddings_type,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config,
    const std::optional<uint32_t>& pad_token,
    const std::optional<Tensor>& optional_output_tensor) {
    using OperationType = EmbeddingsDeviceOperation;
    auto memory_config = output_mem_config.value_or(input_tensor_arg.memory_config());
    auto operation_attributes = OperationType::operation_attributes_t{
        .output_mem_config = memory_config,
        .tilized = tilized,
        .embeddings_type = embeddings_type,
        .pad_token = pad_token,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor_arg = input_tensor_arg,
        .weight_arg = weight_arg,
        .optional_output_tensor = optional_output_tensor,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
