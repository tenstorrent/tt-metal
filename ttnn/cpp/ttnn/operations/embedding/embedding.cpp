// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/embedding/embedding.hpp"

#include <utility>
#include "ttnn/operations/core/core.hpp"
#include "device/embedding_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include <ttnn/operations/copy/typecast/typecast.hpp>

namespace ttnn::operations::embedding {

ttnn::Tensor EmbeddingOperation::invoke(
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
    if (input_tensor.padded_shape()[-1] % tt::constants::TILE_HEIGHT == 0 &&
        weight.padded_shape()[-1] % tt::constants::TILE_WIDTH == 0) {
        if (layout.has_value()) {
            if (layout.value() == ttnn::TILE_LAYOUT) {
                fused_tilized = true;
            }
        } else if (weight_arg.layout() == ttnn::TILE_LAYOUT) {
            fused_tilized = true;
        } else {
            bool typecast_needed = dtype.has_value() && (dtype.value() != weight.dtype());
            TT_FATAL(!typecast_needed, "Can only typecast output embeddings when producing TILE_LAYOUT output");
        }
    }

    auto embeddings = ttnn::prim::embedding(
        input_tensor, weight, fused_tilized, embeddings_type, memory_config, pad_token, optional_output_tensor);
    // Don't include batch_size if there was none
    if (input_tensor_arg.logical_shape().rank() == 1) {
        embeddings = ttnn::reshape(embeddings, Shape({sentence_size, hidden_embedding_dim}));
    } else {
        embeddings = ttnn::reshape(embeddings, Shape({batch_size, sentence_size, hidden_embedding_dim}));
    }
    embeddings = ttnn::to_layout(embeddings, layout.value_or(weight_arg.layout()));
    if (embeddings.layout() == ttnn::TILE_LAYOUT && embeddings.dtype() != dtype.value_or(weight.dtype())) {
        embeddings = ttnn::typecast(embeddings, dtype.value_or(weight.dtype()));
    }
    return embeddings;
}

}  // namespace ttnn::operations::embedding
