// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::transformer {

tt::tt_metal::operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_decode(
    const Tensor& input_tensor,
    const uint32_t num_q_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim,
    const bool overlap_qk_coregrid,
    const bool input_on_subcoregrids,
    const std::optional<const Tensor>& batch_offset,
    std::optional<const uint32_t> slice_size,
    std::vector<Tensor>& output,
    CoreCoord compute_with_storage_grid_size);
tt::tt_metal::operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_decode_interleaved_input(
    const Tensor& input_tensor,
    const uint32_t num_q_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim,
    std::vector<Tensor>& output,
    CoreCoord compute_with_storage_grid_size);
tt::tt_metal::operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_decode_sharded_input(
    const Tensor& input_tensor,
    const uint32_t num_q_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim,
    const bool overlap_qk_coregrid,
    const std::optional<const Tensor>& batch_offset,
    std::optional<const uint32_t> slice_size,
    std::vector<Tensor>& output,
    CoreCoord compute_with_storage_grid_size);
tt::tt_metal::operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_decode_sharded_input_subcoregrid(
    const Tensor& input_tensor,
    const uint32_t num_q_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim,
    const bool overlap_qk_coregrid,
    const std::optional<const Tensor>& batch_offset,
    std::optional<const uint32_t> slice_size,
    std::vector<Tensor>& output,
    CoreCoord compute_with_storage_grid_size);

struct NLPCreateHeadsDecodeDeviceOperation {
    const uint32_t num_q_heads;
    const uint32_t num_kv_heads;
    const uint32_t head_dim;
    const bool overlap_qk_coregrid;
    const bool input_on_subcoregrids;
    std::optional<const uint32_t> slice_size;
    MemoryConfig output_mem_config;

    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;
    static constexpr auto attribute_names = std::forward_as_tuple(
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "overlap_qk_coregrid",
        "input_on_subcoregrids",
        "slice_size",
        "output_mem_config");
    auto attribute_values() const {
        return std::forward_as_tuple(
            this->num_q_heads,
            this->num_kv_heads,
            this->head_dim,
            this->overlap_qk_coregrid,
            this->input_on_subcoregrids,
            this->slice_size,
            this->output_mem_config);
    }
};
}  // namespace ttnn::operations::experimental::transformer
