// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/embedding/device/embedding_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {

namespace operations {

namespace embedding {

struct EmbeddingOperation {
    static inline Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor_arg,
        const Tensor& weight_arg,
        const std::optional<int>& pad_token = std::nullopt,
        const Layout& layout = ttnn::ROW_MAJOR_LAYOUT,
        EmbeddingsType embeddings_type = EmbeddingsType::GENERIC,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        if (pad_token.has_value()) {
            embeddings_type = EmbeddingsType::PADDED;
        }

        auto hidden_embedding_dim = weight_arg.get_shape()[-1];
        auto padded_hidden_embedding_dim = weight_arg.get_shape().with_tile_padding()[-1];
        auto weight = ttnn::unsqueeze_to_4D(weight_arg);

        auto batch_size = input_tensor_arg.get_shape()[0];
        auto sentence_size = input_tensor_arg.get_shape()[-1];
        auto input_tensor =
            ttnn::reshape(input_tensor_arg, ttnn::SimpleShape{std::array<uint32_t, 4>{batch_size, 1, 1, sentence_size}});

        bool tilized = layout == ttnn::TILE_LAYOUT;
        auto embeddings = operation::run(
                              Embeddings{
                                  .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
                                  .tilized = tilized,
                                  .embeddings_type = embeddings_type,
                                  .pad_token = pad_token,
                                  .output_dtype = dtype.value_or(weight.get_dtype())},
                              {input_tensor, weight})
                              .at(0);
        embeddings = ttnn::reshape(
            embeddings, ttnn::SimpleShape{std::array<uint32_t, 3>{batch_size, sentence_size, hidden_embedding_dim}});
        return embeddings;
    }

    static inline auto invoke(
        const Tensor& input_tensor_arg,
        const Tensor& weight_arg,
        const std::optional<int>& pad_token = std::nullopt,
        const Layout& layout = ttnn::ROW_MAJOR_LAYOUT,
        EmbeddingsType embeddings_type = EmbeddingsType::GENERIC,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt
        ) {
            return invoke(DefaultQueueId, input_tensor_arg, weight_arg, pad_token, layout, embeddings_type, dtype, memory_config, optional_output_tensor);
        }
};

}  // namespace embedding
}  // namespace operations

constexpr auto embedding = ttnn::register_operation_with_auto_launch_op<"ttnn::embedding", ttnn::operations::embedding::EmbeddingOperation>();

}  // namespace ttnn
