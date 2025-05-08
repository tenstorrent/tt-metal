// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "split_query_key_value_and_split_heads.hpp"

#include "ttnn/operations/core/core.hpp"

#include "cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.hpp"
#include "cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_falcon7b/nlp_create_qkv_heads_falcon7b.hpp"
#include "cpp/ttnn/operations/experimental/transformer/create_qkv_heads/create_qkv_heads.hpp"

#include "cpp/ttnn/operations/experimental/reshape/view.hpp"

namespace ttnn::operations::transformer {

namespace detail {
std::tuple<Tensor, Tensor, Tensor> reshape_outputs_of_split_query_key_value_and_split_heads(
    const std::tuple<Tensor, Tensor, Tensor>& outputs,
    const uint32_t sequence_size,
    const uint32_t sequence_size_padded,
    const bool transpose_key) {
    auto [query, key, value] = outputs;

    auto batch_size = query.get_logical_shape()[0];
    auto num_heads = query.get_logical_shape()[1];
    auto head_size = query.get_logical_shape()[-1];
    auto head_size_padded = query.get_padded_shape()[-1];

    auto num_kv_heads = value.get_logical_shape()[1];

    query = ttnn::reshape(
        query,
        ttnn::Shape({batch_size, num_heads, sequence_size, head_size}),
        ttnn::Shape({batch_size, num_heads, sequence_size_padded, head_size_padded}));

    if (transpose_key) {
        key = ttnn::reshape(
            key,
            ttnn::Shape({batch_size, num_kv_heads, head_size, sequence_size}),
            ttnn::Shape({batch_size, num_kv_heads, head_size_padded, sequence_size_padded}));
    } else {
        key = ttnn::reshape(
            key,
            ttnn::Shape({batch_size, num_kv_heads, sequence_size, head_size}),
            ttnn::Shape({batch_size, num_kv_heads, sequence_size_padded, head_size_padded}));
    }

    value = ttnn::reshape(
        value,
        ttnn::Shape({batch_size, num_kv_heads, sequence_size, head_size}),
        ttnn::Shape({batch_size, num_kv_heads, sequence_size_padded, head_size_padded}));
    return {query, key, value};
}
}  // namespace detail

std::tuple<Tensor, Tensor, Tensor> SplitQueryKeyValueAndSplitHeadsOperation::invoke(
    const Tensor& input_tensor,
    const std::optional<Tensor>& input_tensor_kv,
    const uint32_t num_heads,
    const std::optional<uint32_t> num_kv_heads,
    const bool transpose_key,
    const std::optional<MemoryConfig>& memory_config) {
    const auto input_shape = input_tensor.get_logical_shape();
    const auto padded_input_shape = input_tensor.get_padded_shape();
    TT_FATAL(input_shape.rank() == 3, "Invalid input tensor: expected 3 dimensions, but found {}.", input_shape.rank());

    TT_FATAL(
        input_tensor.get_layout() == tt::tt_metal::Layout::TILE,
        "Invalid layout: input tensor must use TILE_LAYOUT, but found {}.",
        static_cast<int>(input_tensor.get_layout()));

    TT_FATAL(
        input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
        "Invalid storage type: input tensor must be on a device, but found {}.",
        static_cast<int>(input_tensor.storage_type()));

    const uint32_t sequence_size = input_shape[1];
    const uint32_t sequence_size_padded = padded_input_shape[1];

    if (num_kv_heads.has_value()) {
        TT_FATAL(
            !transpose_key,
            "Invalid configuration: Transpose is set to true, but this is not supported when separate num_kv_heads is "
            "used.");

        uint32_t qkv_heads_times_head_dim = input_shape[2];
        uint32_t qkv_heads_times_head_dim_padded = padded_input_shape[2];
        auto head_size = qkv_heads_times_head_dim / (num_heads + (num_kv_heads.value() * 2));
        auto padded_head_size = qkv_heads_times_head_dim_padded / (num_heads + (num_kv_heads.value() * 2));

        TT_FATAL(
            head_size % TILE_SIZE == 0,
            "Invalid head size: {} is not a multiple of tile size {}. Update the preceding matmul to include padding "
            "in the weights.",
            head_size,
            TILE_SIZE);

        TT_FATAL(
            padded_head_size == head_size,
            "Invalid padding: Head size {} should not have additional tile padding, but padded size is {}.",
            head_size,
            padded_head_size);

        const auto input_4d = ttnn::experimental::view(
            input_tensor, ttnn::Shape{padded_input_shape[0], 1, padded_input_shape[1], padded_input_shape[2]});

        auto outputs = ttnn::experimental::nlp_create_qkv_heads_falcon7b(
            input_4d, memory_config.value_or(input_tensor.memory_config()));
        return detail::reshape_outputs_of_split_query_key_value_and_split_heads(
            {std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs)},
            sequence_size,
            sequence_size_padded,
            transpose_key);
    }

    uint32_t hidden_dim_padded = 0, hidden_dim = 0;
    if (input_tensor_kv.has_value()) {
        const auto input_shape_kv = input_tensor_kv.value().get_logical_shape();
        TT_FATAL(
            input_shape_kv[0] == input_shape[0],
            "Dimension mismatch: KV tensor batch dimension ({}) must match Q tensor batch dimension ({}).",
            input_shape_kv[0],
            input_shape[0]);

        TT_FATAL(
            input_shape_kv[1] == input_shape[1],
            "Dimension mismatch: KV tensor sequence length ({}) must match Q tensor sequence length ({}).",
            input_shape_kv[1],
            input_shape[1]);

        TT_FATAL(
            input_shape_kv[2] == 2 * input_shape[2],
            "Dimension mismatch: KV tensor hidden size ({}) must be twice the Q tensor hidden size ({}).",
            input_shape_kv[2],
            2 * input_shape[2]);

        hidden_dim = input_shape[2];
        hidden_dim_padded = padded_input_shape[2];
    } else {
        hidden_dim = input_shape[2];
        hidden_dim_padded = padded_input_shape[2];
    }

    uint32_t head_size = hidden_dim / num_heads;
    uint32_t padded_head_size = hidden_dim_padded / num_heads;
    TT_FATAL(
        head_size % tt::constants::TILE_WIDTH == 0,
        "Invalid head size: {}. The head size must be a multiple of the tile width ({}). Please adjust the dimensions "
        "accordingly.",
        head_size,
        tt::constants::TILE_WIDTH);

    TT_FATAL(
        padded_head_size == head_size,
        "Padding error: Head size {} should not include additional tile padding, but padded head size was found to be "
        "{}. Ensure that no extra padding is applied.",
        head_size,
        padded_head_size);

    if (input_tensor.is_sharded()) {
        TT_FATAL(
            !input_tensor_kv.has_value(),
            "Invalid operation: KV tensor should not be provided when the input tensor is sharded. Please ensure that "
            "the KV tensor is only used in non-sharded configurations.");

        const auto input_tensor_4d = ttnn::experimental::view(
            input_tensor, ttnn::Shape{padded_input_shape[0], 1, padded_input_shape[1], padded_input_shape[2]});
        return detail::reshape_outputs_of_split_query_key_value_and_split_heads(
            ttnn::experimental::create_qkv_heads(
                input_tensor_4d,
                num_heads,
                num_kv_heads.value_or(num_heads),
                transpose_key,
                memory_config.value_or(input_tensor.memory_config())),
            sequence_size,
            sequence_size_padded,
            transpose_key);
    } else {
        const auto input_tensor_4d = ttnn::experimental::view(
            input_tensor, ttnn::Shape{padded_input_shape[0], 1, padded_input_shape[1], padded_input_shape[2]});
        std::optional<Tensor> input_tensor_kv_4d = std::nullopt;
        if (input_tensor_kv.has_value()) {
            auto padded_input_shape_kv = input_tensor_kv.value().get_padded_shape();
            input_tensor_kv_4d = ttnn::experimental::view(
                input_tensor_kv.value(),
                ttnn::Shape{padded_input_shape_kv[0], 1, padded_input_shape_kv[1], padded_input_shape_kv[2]});
        }
        const auto outputs = ttnn::experimental::nlp_create_qkv_heads(
            input_tensor_4d,
            input_tensor_kv_4d,
            num_heads,
            num_kv_heads.value_or(num_heads),
            transpose_key,
            memory_config.value_or(input_tensor.memory_config()));
        return detail::reshape_outputs_of_split_query_key_value_and_split_heads(
            outputs, sequence_size, sequence_size_padded, transpose_key);
    }
}

}  // namespace ttnn::operations::transformer
