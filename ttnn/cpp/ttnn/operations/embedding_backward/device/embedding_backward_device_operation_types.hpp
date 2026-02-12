// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {

struct EmbeddingBackwardParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INVALID;
    uint32_t num_embeddings = 0;

    static constexpr auto attribute_names =
        std::forward_as_tuple("output_mem_config", "output_dtype", "num_embeddings");
    auto attribute_values() const { return std::forward_as_tuple(output_mem_config, output_dtype, num_embeddings); }
};

struct EmbeddingBackwardInputs {
    Tensor index_tensor;
    Tensor grad_tensor;
    std::optional<Tensor> preallocated_output;

    static constexpr auto attribute_names = std::forward_as_tuple("index_tensor", "grad_tensor", "preallocated_output");
    auto attribute_values() const { return std::forward_as_tuple(index_tensor, grad_tensor, preallocated_output); }
};

}  // namespace ttnn::prim
