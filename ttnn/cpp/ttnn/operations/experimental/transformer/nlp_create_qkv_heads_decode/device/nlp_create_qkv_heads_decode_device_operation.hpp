// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::transformer {

operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_decode(
    const Tensor& input_tensor,
    const uint32_t num_q_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim,
    const bool overlap_qk_coregrid,
    const bool input_on_subcoregrids,
    std::vector<Tensor>& output,
    CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_decode_interleaved_input(
    const Tensor& input_tensor,
    const uint32_t num_q_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim,
    std::vector<Tensor>& output,
    CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_decode_sharded_input(
    const Tensor& input_tensor,
    const uint32_t num_q_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim,
    const bool overlap_qk_coregrid,
    std::vector<Tensor>& output,
    CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_decode_sharded_input_subcoregrid(
    const Tensor& input_tensor,
    const uint32_t num_q_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim,
    const bool overlap_qk_coregrid,
    std::vector<Tensor>& output,
    CoreCoord compute_with_storage_grid_size);

struct NLPCreateHeadsDecodeDeviceOperation {
    const uint32_t num_q_heads;
    const uint32_t num_kv_heads;
    const uint32_t head_dim;
    const bool overlap_qk_coregrid;
    const bool input_on_subcoregrids;
    MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};
}  // namespace ttnn::operations::experimental::transformer
