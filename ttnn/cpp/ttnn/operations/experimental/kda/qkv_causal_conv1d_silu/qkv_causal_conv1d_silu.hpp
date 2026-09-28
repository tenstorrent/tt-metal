// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::kda {

struct QkvCausalConv1dSiluProgramConfig {
    uint32_t channel_chunk_size;
};

// The input layout selects the path:
//  - ROW_MAJOR input: history (ROW_MAJOR) and program_config are required.
//  - TILE input: history is TILE [1,3,Q+K+V] or std::nullopt (three zero rows); program_config is
//    optional (channel_chunk_size = 32 * B, default B = 4). QkvCausalConv1dSiluTiledProgramFactory;
//    q/k/v are bit-identical to the ROW_MAJOR path for the same input, history and taps.
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> qkv_causal_conv1d_silu(
    const ttnn::Tensor& input,
    const std::optional<ttnn::Tensor>& history,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const std::optional<QkvCausalConv1dSiluProgramConfig>& program_config = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

// TILE input only. Also returns new_state = TILE [1,3,Q+K+V] with rows 0-2 = x[T-3..T-1] and
// zero padding rows. new_state is always DRAM interleaved (memory_config applies to q/k/v only).
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> qkv_causal_conv1d_silu_with_conv_state(
    const ttnn::Tensor& input,
    const std::optional<ttnn::Tensor>& history,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const std::optional<QkvCausalConv1dSiluProgramConfig>& program_config = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental::kda
