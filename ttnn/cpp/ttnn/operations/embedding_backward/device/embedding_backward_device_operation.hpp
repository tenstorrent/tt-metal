// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;

namespace ttnn::operations::embedding_backward {

namespace detail {
operation::ProgramWithCallbacks embedding_backward_multi_core(
    const Tensor &index_tensor, const Tensor &grad_tensor, Tensor &output, const uint32_t num_embeddings);
}

struct EmbeddingBackward {
    MemoryConfig output_mem_config;
    DataType output_dtype;
    uint32_t num_embeddings;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

}  // namespace ttnn::operations::embedding_backward
