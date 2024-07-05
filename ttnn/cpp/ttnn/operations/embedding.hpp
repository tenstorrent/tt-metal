// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/experimental//tt_dnn/op_library/embeddings/embeddings_op.hpp"
#include "ttnn/experimental//tt_dnn/op_library/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core.hpp"

namespace ttnn {

namespace operations {

namespace embedding {

using EmbeddingsType = tt::tt_metal::EmbeddingsType;

struct Embedding {
    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor_arg,
        const Tensor& weight_arg,
        const std::optional<int>& pad_token = std::nullopt,
        const Layout& layout = ttnn::ROW_MAJOR_LAYOUT,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        auto embeddings_type = EmbeddingsType::GENERIC;
        if (pad_token.has_value()) {
            embeddings_type = EmbeddingsType::PADDED;
        }

        auto hidden_embedding_dim = weight_arg.get_shape()[-1];
        auto padded_hidden_embedding_dim = weight_arg.get_shape().with_tile_padding()[-1];
        auto weight = ttnn::unsqueeze_to_4D(weight_arg);

        auto batch_size = input_tensor_arg.get_shape()[0];
        auto sentence_size = input_tensor_arg.get_shape()[-1];
        auto input_tensor = ttnn::reshape(input_tensor_arg, ttnn::Shape{{batch_size, 1, 1, sentence_size}});

        bool tilized = layout == ttnn::TILE_LAYOUT;
        auto embeddings = operation::run(
                              tt::tt_metal::Embeddings{
                                  .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
                                  .tilized = tilized,
                                  .embeddings_type = embeddings_type,
                                  .pad_token = pad_token,
                                  .output_dtype = weight.get_dtype()},
                              {input_tensor, weight})
                              .at(0);
        embeddings = ttnn::reshape(embeddings, ttnn::Shape{{batch_size, sentence_size, hidden_embedding_dim}});
        return embeddings;
    }
};

}  // namespace embedding
}  // namespace operations

constexpr auto embedding = ttnn::register_operation<ttnn::operations::embedding::Embedding>("ttnn::embedding");

}  // namespace ttnn
