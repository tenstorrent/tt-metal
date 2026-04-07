// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/embedding/embedding.hpp"

#include <utility>
#include "ttnn/operations/core/core.hpp"
#include "device/embedding_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <tt_stl/small_vector.hpp>

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
    if (pad_token.has_value()) {
        embeddings_type = ttnn::prim::EmbeddingsType::PADDED;
    }
    Tensor mutable_weight = weight_arg;

    if (mutable_weight.layout() == ttnn::TILE_LAYOUT) {
        mutable_weight = ttnn::to_layout(mutable_weight, ttnn::ROW_MAJOR_LAYOUT);
    }
    auto hidden_embedding_dim = mutable_weight.logical_shape()[-1];
    const auto& input_shape = input_tensor.logical_shape();
    auto original_input_rank = input_shape.rank();

    TT_FATAL(original_input_rank >= 1, "Embedding input must have rank >= 1, got rank {}", original_input_rank);
    TT_FATAL(input_shape[-1] > 0, "Last dimension of embedding input must be > 0");

    auto weight = ttnn::unsqueeze_to_4D(mutable_weight);

    // Compute batch_size as product of all dimensions except the last (sequence dimension)
    // This correctly handles ND inputs like [B, 1, 1, S] or [d1, d2, ..., dn]
    uint64_t batch_size = 1;
    for (size_t i = 0; i < original_input_rank - 1; ++i) {
        batch_size *= input_shape[i];
    }
    // For rank 1, batch_size remains 1

    TT_FATAL(batch_size <= std::numeric_limits<uint32_t>::max(), "Batch size overflow: {}", batch_size);

    uint32_t batch_size_u32 = static_cast<uint32_t>(batch_size);
    auto sentence_size = input_shape[-1];

    auto embedding_input_tensor = input_tensor;
    if (input_tensor.layout() == ttnn::ROW_MAJOR_LAYOUT &&
        !(original_input_rank == 4 &&
          input_shape[1] == 1 &&
          input_shape[2] == 1)) {
        embedding_input_tensor = ttnn::reshape(input_tensor, ttnn::Shape({batch_size_u32, 1, 1, sentence_size}));
    }

    // If layout is row major, OR if the input tensor is not a multiple of TILE_HEIGHT, then we cannot use tilized
    bool fused_tilized = false;
    if (embedding_input_tensor.padded_shape()[-1] % tt::constants::TILE_HEIGHT == 0 &&
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
        embedding_input_tensor,
        weight,
        fused_tilized,
        embeddings_type,
        memory_config,
        pad_token,
        optional_output_tensor);

    // Restore original shape with embedding dimension appended
    // Input: [d1, d2, ..., dn] -> Output: [d1, d2, ..., dn, embedding_dim]
    ttsl::SmallVector<uint32_t> output_shape_vec;
    output_shape_vec.reserve(original_input_rank + 1);
    for (size_t i = 0; i < original_input_rank; ++i) {
        output_shape_vec.push_back(input_shape[i]);
    }
    output_shape_vec.push_back(hidden_embedding_dim);
    embeddings = ttnn::reshape(embeddings, ttnn::Shape(std::move(output_shape_vec)));

    embeddings = ttnn::to_layout(embeddings, layout.value_or(weight_arg.layout()));
    if (embeddings.layout() == ttnn::TILE_LAYOUT && embeddings.dtype() != dtype.value_or(weight.dtype())) {
        embeddings = ttnn::typecast(embeddings, dtype.value_or(weight.dtype()));
    }
    return embeddings;
}

}  // namespace ttnn
