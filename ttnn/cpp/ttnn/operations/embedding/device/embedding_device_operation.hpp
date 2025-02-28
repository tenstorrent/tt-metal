// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::constants;

namespace ttnn::operations::embedding {

enum class EmbeddingsType { GENERIC, PADDED, BINARY };
enum class EmbeddingsIndexType { UINT32, BFP16 };

struct Embeddings {
    const tt::tt_metal::MemoryConfig output_mem_config;
    const bool tilized;
    const EmbeddingsType embeddings_type;
    const std::optional<uint32_t> pad_token;
    const tt::tt_metal::DataType output_dtype;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor> &input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
};
}  // namespace ttnn::operations::embedding
