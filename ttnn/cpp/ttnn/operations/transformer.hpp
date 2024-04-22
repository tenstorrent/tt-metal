// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This is a place holder for when the cpp/ttnn folder structure and ttnn namespace is moved over to tt_eager.
#include "tt_dnn/op_library/nlp_tms/nlp_tms.hpp"
#include "tt_dnn/op_library/softmax/softmax_op.hpp"
#include "tensor/tensor.hpp"

#include "tt_dnn/op_library/run_operation.hpp"
#include "ttnn/operations/core.hpp"

namespace ttnn {
namespace operations {
namespace transformer {

inline std::tuple<Tensor, Tensor, Tensor> split_query_key_value_and_split_heads(const Tensor &input_tensor, const std::optional<Tensor> &input_tensor_kv,
    const uint32_t num_heads, const std::optional<uint32_t> num_kv_heads,
    const bool transpose_k_heads,
    const std::optional<MemoryConfig>& memory_config) {
    const auto input_shape = input_tensor.get_legacy_shape();
    TT_FATAL(input_shape.rank() == 3, "Input Tensor must have strictly 3 dimensions!");
    TT_FATAL(input_tensor.get_layout() == tt::tt_metal::Layout::TILE,"Input Tensor must be in a TILE_LAYOUT!");
    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,  "Input_tensor must be on device!");

    if (num_kv_heads.has_value()) {
        TT_FATAL(transpose_k_heads == false, "Transpose = true and separate num_kv_heads is not supported");
        uint32_t qkv_heads_times_head_dim = input_shape[2], qkv_heads_times_head_dim_unpadded = input_shape.without_padding()[2];
        uint32_t head_size = qkv_heads_times_head_dim/(num_heads + (num_kv_heads.value()*2));
        uint32_t unpadded_head_size = qkv_heads_times_head_dim_unpadded /(num_heads + (num_kv_heads.value()*2));

        TT_FATAL(head_size % TILE_SIZE == 0, fmt::format("Head size {} must be a multiple of tile size {}! Update the preceding matmul to have the padding in the weights!", head_size, TILE_WIDTH));
        TT_FATAL(unpadded_head_size == head_size, fmt::format("Head size {} cannot have tile padding", head_size));

        const auto input_4d = input_tensor.reshape(input_shape[0], 1, input_shape[1], input_shape[2]);
        auto outputs = tt::tt_metal::nlp_create_qkv_heads_falcon7b(input_4d, memory_config.value_or(input_tensor.memory_config()));
        return {outputs.at(0), outputs.at(1), outputs.at(2)};
    }

    uint32_t hidden_dim_padded = 0, hidden_dim = 0;
    if (input_tensor_kv.has_value()) {
        const auto input_shape_kv = input_tensor_kv.value().get_legacy_shape();
        TT_FATAL(input_shape_kv[0] == input_shape[0], "KV tensor batch dim must be same as Q tensor batch!");
        TT_FATAL(input_shape_kv[1] == input_shape[1], "KV tensor seq_len dim must be same as Q tensor seq_len!");
        TT_FATAL(input_shape_kv[2] == 2*input_shape[2], "KV tensor hidden size must be 2 times Q hidden size");
        hidden_dim_padded = input_shape[2];
        hidden_dim = input_shape.without_padding()[2];
    }
    else {
        hidden_dim_padded = input_shape[2];
        hidden_dim = input_shape.without_padding()[2];
    }

    uint32_t head_size = hidden_dim / num_heads;
    uint32_t padded_head_size = hidden_dim_padded / num_heads;
    TT_FATAL(head_size % TILE_WIDTH == 0, fmt::format("Head size {} must be a multiple of tile width {}", head_size, TILE_WIDTH));
    TT_FATAL(padded_head_size == head_size, fmt::format("Head size {} cannot have tile padding", head_size));

    if (input_tensor.is_sharded()) {
        TT_FATAL(input_tensor_kv.has_value() == false, "KV tensor cannot be passed in when sharded");
        const auto input_tensor_4d = input_tensor.reshape(input_shape[0], 1, input_shape[1], input_shape[2]);
        return tt::tt_metal::create_qkv_heads(input_tensor_4d, num_heads, num_kv_heads.value_or(num_heads), transpose_k_heads, memory_config.value_or(input_tensor.memory_config()));
    }
    else {
        const auto input_tensor_4d = input_tensor.reshape(input_shape[0], 1, input_shape[1], input_shape[2]);
        std::optional<Tensor> input_tensor_kv_4d = std::nullopt;
        if (input_tensor_kv.has_value()) {
            auto input_shape_kv = input_tensor_kv.value().get_legacy_shape();
            input_tensor_kv_4d = input_tensor_kv.value().reshape(input_shape_kv[0], 1, input_shape_kv[1], input_shape_kv[2]);
        }
        const auto outputs = tt::tt_metal::nlp_create_qkv_heads(input_tensor_4d, input_tensor_kv_4d, num_heads, num_kv_heads.value_or(num_heads), transpose_k_heads, memory_config.value_or(input_tensor.memory_config()));
        return {outputs.at(0), outputs.at(1), outputs.at(2)};
    }
}
}  // namespace transformer
}  // namespace operations
}  // namespace ttnn
