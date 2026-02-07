// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {

enum class EmbeddingsType { GENERIC, PADDED, BINARY };
enum class EmbeddingsIndexType { UINT32, BFP16 };

struct EmbeddingParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    bool tilized = false;
    EmbeddingsType embeddings_type = EmbeddingsType::GENERIC;
    std::optional<uint32_t> pad_token;

    static constexpr auto attribute_names =
        std::forward_as_tuple("output_mem_config", "tilized", "embeddings_type", "pad_token");
    auto attribute_values() const {
        return std::forward_as_tuple(output_mem_config, tilized, embeddings_type, pad_token);
    }
};

struct EmbeddingInputs {
    Tensor input_tensor_arg;
    Tensor weight_arg;
    std::optional<Tensor> optional_output_tensor;

    static constexpr auto attribute_names =
        std::forward_as_tuple("input_tensor_arg", "weight_arg", "optional_output_tensor");
    auto attribute_values() const {
        return std::forward_as_tuple(input_tensor_arg, weight_arg, optional_output_tensor);
    }
};

}  // namespace ttnn::prim
