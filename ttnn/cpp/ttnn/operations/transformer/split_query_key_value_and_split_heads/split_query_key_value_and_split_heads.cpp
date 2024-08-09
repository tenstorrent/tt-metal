// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "split_query_key_value_and_split_heads.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/nlp_tms/nlp_tms.hpp"

#include "ttnn/operations/core/core.hpp"

#include "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.hpp"


namespace ttnn::operations::transformer {

namespace detail {
std::tuple<Tensor, Tensor, Tensor> reshape_outputs_of_split_query_key_value_and_split_heads(
    const std::tuple<Tensor, Tensor, Tensor>& outputs,
    const uint32_t sequence_size,
    const uint32_t sequence_size_padded,
    const bool transpose_key) {
    auto [query, key, value] = outputs;

    auto batch_size = query.get_shape()[0];
    auto num_heads = query.get_shape()[1];
    auto head_size = query.get_shape()[-1];
    auto head_size_padded = query.get_shape().with_tile_padding()[-1];

    auto num_kv_heads = value.get_shape()[1];

    query = ttnn::reshape(
        query,
        ttnn::Shape(tt::tt_metal::Shape(
            std::array{batch_size, num_heads, sequence_size, head_size},
            std::array{batch_size, num_heads, sequence_size_padded, head_size_padded})));

    if (transpose_key) {
        key = ttnn::reshape(
            key,
            ttnn::Shape(tt::tt_metal::Shape(
                std::array{batch_size, num_kv_heads, head_size, sequence_size},
                std::array{batch_size, num_kv_heads, head_size_padded, sequence_size_padded})));
    } else {
        key = ttnn::reshape(
            key,
            ttnn::Shape(tt::tt_metal::Shape(
                std::array{batch_size, num_kv_heads, sequence_size, head_size},
                std::array{batch_size, num_kv_heads, sequence_size_padded, head_size_padded})));
    }

    value = ttnn::reshape(
        value,
        ttnn::Shape(tt::tt_metal::Shape(
            std::array{batch_size, num_kv_heads, sequence_size, head_size},
            std::array{batch_size, num_kv_heads, sequence_size_padded, head_size_padded})));
    return {query, key, value};
}
}  // namespace detail

std::tuple<Tensor, Tensor, Tensor> SplitQueryKeyValueAndSplitHeadsOperation::operator()(
    const Tensor& input_tensor,
    const std::optional<Tensor>& input_tensor_kv,
    const uint32_t num_heads,
    const std::optional<uint32_t> num_kv_heads,
    const bool transpose_key,
    const std::optional<MemoryConfig>& memory_config) {
    const auto input_shape = input_tensor.get_shape();
    TT_FATAL(input_shape.rank() == 3, "Input Tensor must have strictly 3 dimensions!");
    TT_FATAL(input_tensor.get_layout() == tt::tt_metal::Layout::TILE, "Input Tensor must be in a TILE_LAYOUT!");
    TT_FATAL(
        input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE or
            input_tensor.storage_type() == tt::tt_metal::StorageType::MULTI_DEVICE,
        "Input_tensor must be on device!");

    const uint32_t sequence_size = input_shape[1];
    const uint32_t sequence_size_padded = input_shape.with_tile_padding()[1];

    if (num_kv_heads.has_value()) {
        TT_FATAL(transpose_key == false, "Transpose = true and separate num_kv_heads is not supported");
        uint32_t qkv_heads_times_head_dim = input_shape[2],
                 qkv_heads_times_head_dim_padded = input_shape.with_tile_padding()[2];
        auto head_size = qkv_heads_times_head_dim / (num_heads + (num_kv_heads.value() * 2));
        auto padded_head_size = qkv_heads_times_head_dim_padded / (num_heads + (num_kv_heads.value() * 2));

        TT_FATAL(
            head_size % TILE_SIZE == 0,
            fmt::format(
                "Head size {} must be a multiple of tile size {}! Update the preceding matmul to have the padding in "
                "the weights!",
                head_size,
                TILE_WIDTH));
        TT_FATAL(padded_head_size == head_size, fmt::format("Head size {} cannot have tile padding", head_size));

        const auto input_4d = input_tensor.reshape(
            input_shape.with_tile_padding()[0],
            1,
            input_shape.with_tile_padding()[1],
            input_shape.with_tile_padding()[2]);
        auto outputs =
            tt::tt_metal::nlp_create_qkv_heads_falcon7b(input_4d, memory_config.value_or(input_tensor.memory_config()));
        return detail::reshape_outputs_of_split_query_key_value_and_split_heads(
            {outputs.at(0), outputs.at(1), outputs.at(2)}, sequence_size, sequence_size_padded, transpose_key);
    }

    uint32_t hidden_dim_padded = 0, hidden_dim = 0;
    if (input_tensor_kv.has_value()) {
        const auto input_shape_kv = input_tensor_kv.value().get_shape();
        TT_FATAL(input_shape_kv[0] == input_shape[0], "KV tensor batch dim must be same as Q tensor batch!");
        TT_FATAL(input_shape_kv[1] == input_shape[1], "KV tensor seq_len dim must be same as Q tensor seq_len!");
        TT_FATAL(input_shape_kv[2] == 2 * input_shape[2], "KV tensor hidden size must be 2 times Q hidden size");
        hidden_dim = input_shape[2];
        hidden_dim_padded = input_shape.with_tile_padding()[2];
    } else {
        hidden_dim = input_shape[2];
        hidden_dim_padded = input_shape.with_tile_padding()[2];
    }

    uint32_t head_size = hidden_dim / num_heads;
    uint32_t padded_head_size = hidden_dim_padded / num_heads;
    TT_FATAL(
        head_size % TILE_WIDTH == 0,
        fmt::format("Head size {} must be a multiple of tile width {}", head_size, TILE_WIDTH));
    TT_FATAL(padded_head_size == head_size, fmt::format("Head size {} cannot have tile padding", head_size));

    if (input_tensor.is_sharded()) {
        TT_FATAL(not input_tensor_kv.has_value(), "KV tensor cannot be passed in when sharded");
        const auto input_tensor_4d = input_tensor.reshape(
            input_shape.with_tile_padding()[0],
            1,
            input_shape.with_tile_padding()[1],
            input_shape.with_tile_padding()[2]);
        return detail::reshape_outputs_of_split_query_key_value_and_split_heads(
            tt::tt_metal::create_qkv_heads(
                input_tensor_4d,
                num_heads,
                num_kv_heads.value_or(num_heads),
                transpose_key,
                memory_config.value_or(input_tensor.memory_config())),
            sequence_size,
            sequence_size_padded,
            transpose_key);
    } else {
        const auto input_tensor_4d = input_tensor.reshape(
            input_shape.with_tile_padding()[0],
            1,
            input_shape.with_tile_padding()[1],
            input_shape.with_tile_padding()[2]);
        std::optional<Tensor> input_tensor_kv_4d = std::nullopt;
        if (input_tensor_kv.has_value()) {
            auto padded_input_shape_kv = input_tensor_kv.value().get_shape().with_tile_padding();
            input_tensor_kv_4d = input_tensor_kv.value().reshape(
                padded_input_shape_kv[0], 1, padded_input_shape_kv[1], padded_input_shape_kv[2]);
        }
        const auto outputs = ttnn::experimental::nlp_create_qkv_heads(
            input_tensor_4d,
            input_tensor_kv_4d,
            num_heads,
            num_kv_heads.value_or(num_heads),
            transpose_key,
            memory_config.value_or(input_tensor.memory_config()));
        return detail::reshape_outputs_of_split_query_key_value_and_split_heads(outputs, sequence_size, sequence_size_padded, transpose_key);
    }
}

}  // namespace ttnn::operations::transformer
