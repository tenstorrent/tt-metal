// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/base_types.hpp>
#include <optional>

namespace ttnn::operations::experimental::moe_gpt {

struct operation_attributes_t {
    uint32_t num_experts{};
    uint32_t layer_id{};
    bool enable_dram_output{false};
    std::optional<uint32_t> cluster_axis;
};

struct tensor_args_t {
    const Tensor& input_tensor;
    const Tensor& w0_w1_tensor;
    const Tensor& w2_tensor;
    const Tensor& output_tensor;
    std::optional<Tensor> dram_output_tensor;

    // Tilize phase inputs (all must be present, or none)
    std::optional<Tensor> sparse_buffer;
    std::optional<Tensor> expert_indices;
    std::optional<Tensor> expert_scores;
    std::optional<Tensor> expert_mapping;

    // Pre-allocated tilize output tensor (DRAM, written to by tilize writer)
    std::optional<Tensor> tilize_output;

    // Check if all tilize input tensors are present (NOT tilize_output)
    bool has_tilize_inputs() const {
        return sparse_buffer.has_value() && expert_indices.has_value() && expert_scores.has_value() &&
               expert_mapping.has_value();
    }

    // Check if ALL tilize tensors are present including output (TILIZE_TO_DRAM mode)
    bool has_tilize_args() const { return has_tilize_inputs() && tilize_output.has_value(); }
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::moe_gpt
