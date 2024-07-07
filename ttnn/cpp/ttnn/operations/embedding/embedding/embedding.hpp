// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/operations/embedding/embedding/device/embeddings_op.hpp"
#include "tt_eager/tt_dnn/op_library/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core.hpp"

namespace ttnn {

namespace operations {

namespace embedding {

struct Embedding {
    static inline Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& input_tensor_arg,
        const Tensor& weight_arg,
        const std::optional<int>& pad_token = std::nullopt,
        const Layout& layout = ttnn::ROW_MAJOR_LAYOUT,
        EmbeddingsType embeddings_type = EmbeddingsType::GENERIC,
        const std::optional<const DataType> output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt
        ) {

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
                              Embeddings{
                                  .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
                                  .tilized = tilized,
                                  .embeddings_type = embeddings_type,
                                  .pad_token = pad_token,
                                  .output_dtype = output_dtype.value_or(weight.get_dtype())},
                              {input_tensor, weight})
                              .at(0);
        embeddings = ttnn::reshape(embeddings, ttnn::Shape{{batch_size, sentence_size, hidden_embedding_dim}});
        return embeddings;
    }

    static inline auto execute_on_worker_thread(
        const Tensor& input_tensor_arg,
        const Tensor& weight_arg,
        const std::optional<int>& pad_token = std::nullopt,
        const Layout& layout = ttnn::ROW_MAJOR_LAYOUT,
        EmbeddingsType embeddings_type = EmbeddingsType::GENERIC,
        const std::optional<const DataType> output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt
        ) {
            constexpr auto DefaultQueueId = 0;
            return execute_on_worker_thread(DefaultQueueId, input_tensor_arg, weight_arg, pad_token, layout, embeddings_type, output_dtype, memory_config, optional_output_tensor);
        }
};

}  // namespace embedding
}  // namespace operations

constexpr auto embedding = ttnn::register_operation<ttnn::operations::embedding::Embedding>("ttnn::embedding");

}  // namespace ttnn
