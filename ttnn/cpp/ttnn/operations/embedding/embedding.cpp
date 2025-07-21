// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/embedding/embedding.hpp"

#include <utility>
#include "ttnn/operations/core/core.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/embedding/device/embedding_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"

namespace ttnn::operations::embedding {

ttnn::Tensor EmbeddingOperation::invoke(
    QueueId queue_id,
    const Tensor& input_tensor_arg,
    const Tensor& weight_arg,
    const std::optional<int>& pad_token,
    const std::optional<ttnn::Layout>& layout,
    EmbeddingsType embeddings_type,
    const std::optional<const DataType> dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    if (pad_token.has_value()) {
        embeddings_type = EmbeddingsType::PADDED;
    }
    Tensor mutable_input_tensor = input_tensor_arg;
    Tensor mutable_weight = weight_arg;

    if (mutable_weight.layout() == ttnn::TILE_LAYOUT) {
        mutable_weight = ttnn::to_layout(mutable_weight, ttnn::ROW_MAJOR_LAYOUT);
    }
    auto hidden_embedding_dim = mutable_weight.logical_shape()[-1];
    auto padded_hidden_embedding_dim = mutable_weight.padded_shape()[-1];
    auto weight = ttnn::unsqueeze_to_4D(mutable_weight);

    // If indices tensor is 1 dimensional, batch size is 1
    auto batch_size = (mutable_input_tensor.logical_shape().rank() == 1) ? 1 : mutable_input_tensor.logical_shape()[0];
    auto sentence_size = mutable_input_tensor.logical_shape()[-1];
    auto input_tensor = mutable_input_tensor;
    if (mutable_input_tensor.layout() == ttnn::ROW_MAJOR_LAYOUT) {
        input_tensor = ttnn::reshape(mutable_input_tensor, ttnn::Shape({batch_size, 1, 1, sentence_size}));
    }

    // If layout is row major, OR if the input tensor is not a multiple of TILE_HEIGHT, then we cannot use tilized
    bool fused_tilized = false;
    if (input_tensor.padded_shape()[-1] % TILE_HEIGHT == 0 && weight.padded_shape()[-1] % TILE_WIDTH == 0) {
        if (layout.has_value()) {
            if (layout.value() == ttnn::TILE_LAYOUT) {
                fused_tilized = true;
            }
        } else if (weight_arg.layout() == ttnn::TILE_LAYOUT) {
            fused_tilized = true;
        }
    }

    auto embeddings = tt::tt_metal::operation::run(
                          Embeddings{
                              .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
                              .tilized = fused_tilized,
                              .embeddings_type = embeddings_type,
                              .pad_token = pad_token,
                              .output_dtype = dtype.value_or(weight.dtype())},
                          {input_tensor, weight})
                          .at(0);
    // Don't include batch_size if there was none
    if (input_tensor_arg.logical_shape().rank() == 1) {
        embeddings = ttnn::reshape(embeddings, Shape({sentence_size, hidden_embedding_dim}));
    } else {
        embeddings = ttnn::reshape(embeddings, Shape({batch_size, sentence_size, hidden_embedding_dim}));
    }
    embeddings = ttnn::to_layout(embeddings, layout.value_or(weight_arg.layout()));
    return embeddings;
}
ttnn::Tensor EmbeddingOperation::invoke(
    const Tensor& input_tensor_arg,
    const Tensor& weight_arg,
    const std::optional<int>& pad_token,
    const std::optional<ttnn::Layout>& layout,
    EmbeddingsType embeddings_type,
    const std::optional<const DataType> dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return invoke(
        DefaultQueueId,
        input_tensor_arg,
        weight_arg,
        pad_token,
        layout,
        embeddings_type,
        dtype,
        memory_config,
        std::move(optional_output_tensor));
}

}  // namespace ttnn::operations::embedding
