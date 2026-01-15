// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::embedding {

enum class EmbeddingsType { GENERIC, PADDED, BINARY };
enum class EmbeddingsIndexType { UINT32, BFP16 };

struct EmbeddingParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    bool tilized = false;
    EmbeddingsType embeddings_type = EmbeddingsType::GENERIC;
    std::optional<uint32_t> pad_token;
};

struct EmbeddingInputs {
    Tensor input_tensor_arg;
    Tensor weight_arg;
    std::optional<Tensor> optional_output_tensor;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::embedding
