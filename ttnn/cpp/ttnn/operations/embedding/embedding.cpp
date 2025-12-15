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

namespace ttnn {

ttnn::Tensor embedding(
    const Tensor& input_tensor,
    const Tensor& weight_arg,
    const std::optional<int>& pad_token,
    const std::optional<ttnn::Layout>& layout,
    ttnn::prim::EmbeddingsType embeddings_type,
    const std::optional<const DataType> dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    TT_FATAL(
        input_tensor_arg.logical_shape().rank() <= 2,
        "EmbeddingOp only supports input tensors of rank 1 or 2. Got rank = {}, shape = {}",
        input_tensor_arg.logical_shape().rank(),
        input_tensor_arg.logical_shape().to_string());

    if (pad_token.has_value()) {
        embeddings_type = ttnn::prim::EmbeddingsType::PADDED;
    }
    Tensor mutable_weight = weight_arg;

    if (mutable_weight.layout() == ttnn::TILE_LAYOUT) {
        mutable_weight = ttnn::to_layout(mutable_weight, ttnn::ROW_MAJOR_LAYOUT);
    }

    auto weight = ttnn::unsqueeze_to_4D(mutable_weight);

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

    if (input_tensor.logical_shape().size() == 4 && input_tensor.logical_shape()[1] == 1 &&
        input_tensor.logical_shape()[2] == 1) {
        // Although the output shape is expected to match the input shape
        // with an additional (last_dim,) appended from the weight tensor,
        // the current implementation returns 3D shape when given a 4D input.
        //
        // To maintain compatibility with the existing behavior, the following reshape
        // logic is preserved.
        //
        // Example:
        // Input shape:      (batch_size, 1, 1, sequence_length)
        // Output shape:     (batch_size, sequence_length, embedding_dim)
        embeddings = ttnn::reshape(
            embeddings,
            ttnn::Shape(
                {embeddings.logical_shape()[0], embeddings.logical_shape()[-2], embeddings.logical_shape()[-1]}));
    }
    embeddings = ttnn::to_layout(embeddings, layout.value_or(weight_arg.layout()));
    if (embeddings.layout() == ttnn::TILE_LAYOUT && embeddings.dtype() != dtype.value_or(weight.dtype())) {
        embeddings = ttnn::typecast(embeddings, dtype.value_or(weight.dtype()));
    }
    return embeddings;
}

}  // namespace ttnn
