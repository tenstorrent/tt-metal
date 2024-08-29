// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::operations::transformer::detail {

operation::ProgramWithCallbacks sdpa_multi_core(
    const Tensor &input_tensor_q,
    const Tensor &input_tensor_k,
    const Tensor &input_tensor_v,
    const Tensor &output_tensor,
    const std::optional<const Tensor> causal_mask,
    std::optional<float> scale,
    bool is_causal,
    std::size_t q_chunk_size,
    std::size_t k_chunk_size,
    DeviceComputeKernelConfig compute_kernel_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<const uint32_t> valid_seq_len);

}  // namespace ttnn::operations::transformer::detail
