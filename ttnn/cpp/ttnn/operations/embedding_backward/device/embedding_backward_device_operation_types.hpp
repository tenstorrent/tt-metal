// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct EmbeddingBackwardParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INVALID;
    uint32_t num_embeddings = 0;
};

struct EmbeddingBackwardInputs {
    Tensor index_tensor;
    Tensor grad_tensor;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::prim
