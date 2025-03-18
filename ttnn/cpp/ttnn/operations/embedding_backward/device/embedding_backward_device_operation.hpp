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
tt::tt_metal::operation::ProgramWithCallbacks embedding_backward_multi_core(
    const Tensor &index_tensor, const Tensor &grad_tensor, Tensor &output, const uint32_t num_embeddings);
}

struct EmbeddingBackward {
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    uint32_t num_embeddings;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor> &input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

}  // namespace ttnn::operations::embedding_backward
