// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/embedding/device/embedding_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/common/hash_without_rows.hpp"
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
            TT_FATAL(
                operation_attributes.output_mem_config.is_l1(),
                "Embedding sharded Row Major output must be in L1: the reader writes rows straight into its own "
                "shard");
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

namespace {
// The fused factory with an interleaved output derives every per-core count from the index length on
// device; only that configuration hashes without the index length and the table row count.
bool row_free_configuration(
    const EmbeddingsDeviceOperation::operation_attributes_t& operation_attributes,
    const EmbeddingsDeviceOperation::tensor_args_t& tensor_args) {
    return operation_attributes.tilized && tensor_args.input_tensor_arg.layout() == Layout::ROW_MAJOR &&
           !is_sharded(operation_attributes.output_mem_config.memory_layout());
}
}  // namespace

void EmbeddingsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input_tensor_arg;
    if (operation_attributes.tilized) {
        TT_FATAL(
            a.padded_shape()[-1] % TILE_HEIGHT == 0,
            "Input tensor width {} must be a multiple of tile height {} to have the output tensor tilized",
            a.padded_shape()[-1],
            TILE_HEIGHT);
    }
    if (a.layout() == Layout::ROW_MAJOR) {
        TT_FATAL(a.padded_shape()[1] == 1 && a.padded_shape()[2] == 1, "Only dim 0 && 3 for the input can be non 1");
    }
    if (tensor_args.optional_output_tensor.has_value()) {
        const auto& output_shape = tensor_args.optional_output_tensor->padded_shape();
        TT_FATAL(
            output_shape[0] == a.padded_shape()[0] && output_shape[-2] == a.padded_shape()[-1],
            "Preallocated output shape {} does not match the index tensor shape {}",
            output_shape,
            a.padded_shape());
    }
}

ttsl::hash::hash_t EmbeddingsDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using ttnn::operations::common::hash_spec_without_rows;
    constexpr uint64_t kSalt = 0x656d6265645f6673ULL;  // "embed_fs"
    const auto& index = tensor_args.input_tensor_arg;
    const auto& weights = tensor_args.weight_arg;
    const bool row_free = row_free_configuration(operation_attributes, tensor_args);
    ttsl::hash::hash_t index_hash;
    ttsl::hash::hash_t weights_hash;
    ttsl::hash::hash_t output_hash = 0;
    if (row_free) {
        // The index length is shape[-1] of a [B, 1, 1, S] row-major tensor: keep everything but S.
        const auto& layout = index.tensor_spec().tensor_layout();
        index_hash = ttsl::hash::hash_objects_with_default_seed(
            kSalt,
            layout.get_data_type(),
            layout.get_page_config(),
            layout.get_memory_config(),
            index.logical_shape().rank(),
            index.padded_shape()[0]);
        weights_hash = hash_spec_without_rows(weights, kSalt);
        if (tensor_args.optional_output_tensor.has_value()) {
            output_hash = hash_spec_without_rows(*tensor_args.optional_output_tensor, kSalt);
        }
    } else {
        index_hash = ttsl::hash::hash_objects_with_default_seed(kSalt, index.tensor_spec());
        weights_hash = ttsl::hash::hash_objects_with_default_seed(kSalt, weights.tensor_spec());
        if (tensor_args.optional_output_tensor.has_value()) {
            output_hash =
                ttsl::hash::hash_objects_with_default_seed(kSalt, tensor_args.optional_output_tensor->tensor_spec());
        }
    }
    return tt::tt_metal::operation::hash_operation<EmbeddingsDeviceOperation>(
        operation_attributes.output_mem_config,
        operation_attributes.tilized,
        static_cast<uint8_t>(operation_attributes.embeddings_type),
        operation_attributes.pad_token,
        row_free,
        index_hash,
        weights_hash,
        tensor_args.optional_output_tensor.has_value(),
        output_hash);
}

tt::tt_metal::TensorSpec EmbeddingsDeviceOperation::compute_output_specs(
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
    return tt::tt_metal::TensorSpec(
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
