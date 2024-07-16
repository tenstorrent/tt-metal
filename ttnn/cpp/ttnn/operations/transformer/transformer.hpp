// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This is a place holder for when the cpp/ttnn folder structure and ttnn namespace is moved over to tt_eager.
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_dnn/op_library/nlp_tms/nlp_tms.hpp"
#include "tt_dnn/op_library/rotary_embedding/rotary_embedding_op.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_op.hpp"
#include "ttnn/operations/core.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"

namespace ttnn {
namespace operations::transformer {

namespace detail {
inline std::tuple<Tensor, Tensor, Tensor> reshape_outputs_of_split_query_key_value_and_split_heads(
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

inline std::tuple<Tensor, Tensor, Tensor> split_query_key_value_and_split_heads(
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
        const auto outputs = tt::tt_metal::nlp_create_qkv_heads(
            input_tensor_4d,
            input_tensor_kv_4d,
            num_heads,
            num_kv_heads.value_or(num_heads),
            transpose_key,
            memory_config.value_or(input_tensor.memory_config()));
        return detail::reshape_outputs_of_split_query_key_value_and_split_heads(
            {outputs.at(0), outputs.at(1), outputs.at(2)}, sequence_size, sequence_size_padded, transpose_key);
    }
}

struct ConcatenateHeads : public tt::tt_metal::NlpConcatHeads {
    void validate(const std::vector<Tensor>& input_tensors) const {
        const auto& input_tensor = input_tensors.at(0);
        const auto head_size = input_tensor.get_shape()[-1];
        const auto padded_head_size = input_tensor.get_legacy_shape()[-1];
        TT_FATAL(
            head_size % ttnn::types::TILE_SIZE == 0,
            fmt::format(
                "Head size must be a multiple of {} but was found to be {}! Update matmul that uses the output of this "
                "operation to have the "
                "padding in the weights!",
                ttnn::types::TILE_SIZE,
                head_size));
        TT_FATAL(padded_head_size - head_size == 0, "Head size cannot have tile padding!");

        NlpConcatHeads::validate(input_tensors);
    }

    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
        std::vector<tt::tt_metal::Shape> output_shape_vec;
        const auto& input_tensor = input_tensors.at(0);
        const ttnn::types::Shape input_shape = input_tensor.get_shape();
        const ttnn::types::Shape padded_input_shape = input_shape.with_tile_padding();

        auto batch_size = input_shape[0];
        auto num_heads = input_shape[1];
        auto sequence_size = input_shape[2];
        auto padded_sequence_size = padded_input_shape[2];
        auto head_size = input_shape[3];
        auto padded_head_size = padded_input_shape[3];

        std::array<uint32_t, 3> intended_output_shape = {batch_size, sequence_size, num_heads * head_size};
        std::array<uint32_t, 3> padded_output_shape = {batch_size, padded_sequence_size, num_heads * padded_head_size};
        return {ttnn::types::Shape(intended_output_shape, padded_output_shape).value()};
    }

    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const {
        const auto& input_tensor = input_tensors.at(0);
        if (this->output_mem_config.is_sharded()) {
            ShardSpec shard_spec = input_tensor.shard_spec().value();
            uint32_t num_cores = shard_spec.num_cores();
            uint32_t heads_per_shard = shard_spec.shape[0] / input_tensor.get_legacy_shape()[-2];
            shard_spec.shape = {shard_spec.shape[0] / heads_per_shard, shard_spec.shape[1] * heads_per_shard};
            auto mem_config = this->output_mem_config;
            mem_config.shard_spec = shard_spec;
            return {create_device_tensor(
                this->compute_output_shapes(input_tensors).at(0),
                input_tensor.get_dtype(),
                Layout::TILE,
                input_tensor.device(),
                mem_config)};
        } else {
            return operation::generic_create_output_tensors(
                *this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
        }
    }
};

struct ExecuteConcatenateHeads {

    static inline ttnn::Tensor execute_on_worker_thread(
        const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
        return operation::run(ConcatenateHeads{memory_config.value_or(input_tensor.memory_config())}, {input_tensor})
            .at(0);
    }
};

struct ExecuteRotaryEmbedding {

    static inline ttnn::Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        const Tensor& cos_cache,
        const Tensor& sin_cache,
        const std::optional<uint32_t> token_index = std::nullopt,
        const std::optional<MemoryConfig> memory_config = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
        uint32_t seq_len = input_tensor.get_legacy_shape()[-2];
        uint32_t B = input_tensor.get_legacy_shape()[0];
        uint32_t X = input_tensor.get_legacy_shape()[-1];

        auto arch = input_tensor.device()->arch();
        auto kernel_config_val =
            init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);

        return operation::run(
                   tt::tt_metal::RotaryEmbedding{
                       seq_len, token_index, memory_config.value_or(input_tensor.memory_config()), kernel_config_val},
                   {input_tensor, cos_cache, sin_cache})
            .at(0);
    }
};

template <bool in_place>
struct ExecuteAttentionSoftmax {

    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        const std::optional<int>& head_size_arg = std::nullopt,
        const std::optional<const ttnn::Tensor>& attention_mask = std::nullopt,
        const ttnn::operations::normalization::SoftmaxProgramConfig& program_config = ttnn::operations::normalization::SoftmaxDefaultProgramConfig{},
        const std::optional<bool> causal_mask = false,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt) {
        float head_size = head_size_arg.has_value() ? 1.0f / std::sqrt(head_size_arg.value()) : 1.0f;
        if constexpr (in_place) {
            TT_FATAL(attention_mask.has_value(), "Cannot apply divide by sqrt(head_size) using in-place version!");
        } else {
            if (not attention_mask.has_value()) {
                auto output_tensor = ttnn::multiply(input_tensor, head_size);
                return ttnn::operations::normalization::softmax(output_tensor, memory_config.value_or(input_tensor.memory_config()));
            }
        }

        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
        auto kernel_config_val = init_device_compute_kernel_config(
            input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, false, false);
        auto output_tensor = operation::run(
                                ttnn::operations::normalization::Softmax{
                                     head_size,
                                     in_place,
                                     memory_config.value_or(input_tensor.memory_config()),
                                     program_config,
                                     causal_mask.value(),
                                     kernel_config_val,
                                     false},
                                 {input_tensor},
                                 {attention_mask})
                                 .at(0);
        return output_tensor;
    }
};

}  // namespace operations::transformer

namespace transformer {

constexpr auto split_query_key_value_and_split_heads = ttnn::register_operation(
    "ttnn::transformer::split_query_key_value_and_split_heads",
    TO_LAMBDA(ttnn::operations::transformer::split_query_key_value_and_split_heads));

constexpr auto concatenate_heads = ttnn::register_operation<ttnn::operations::transformer::ExecuteConcatenateHeads>(
    "ttnn::transformer::concatenate_heads");

constexpr auto rotary_embedding = ttnn::register_operation<ttnn::operations::transformer::ExecuteRotaryEmbedding>(
    "ttnn::transformer::rotary_embedding");

constexpr auto attention_softmax =
    ttnn::register_operation<ttnn::operations::transformer::ExecuteAttentionSoftmax<false>>(
        "ttnn::transformer::attention_softmax");
constexpr auto attention_softmax_ =
    ttnn::register_operation<ttnn::operations::transformer::ExecuteAttentionSoftmax<true>>(
        "ttnn::transformer::attention_softmax_");

}  // namespace transformer

}  // namespace ttnn
