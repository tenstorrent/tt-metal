// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::operations::experimental::transformer::detail {

operation::ProgramWithCallbacks speculative_sdpa_decode_multi_core(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    std::optional<const Tensor> cur_pos_tensor,
    std::optional<const Tensor> page_table_tensor,
    std::optional<const Tensor> attn_mask,
    std::optional<const Tensor> priority_tensor,
    const Tensor& full_output_tensor,
    const Tensor& speculated_output_tensor,
    const Tensor& l2_dist_tensor,
    const Tensor& l2_norm_tensor,
    bool is_causal,
    const std::vector<uint32_t>& cur_pos_ids,
    std::optional<float> scale,
    std::optional<float> lambda,
    DeviceComputeKernelConfig compute_kernel_config,
    std::optional<SDPAProgramConfig> program_config,
    const uint32_t k_chunk_size,
    const uint32_t speculative_chunk_size,
    std::optional<bool> share_cache);

}  // namespace ttnn::operations::experimental::transformer::detail
