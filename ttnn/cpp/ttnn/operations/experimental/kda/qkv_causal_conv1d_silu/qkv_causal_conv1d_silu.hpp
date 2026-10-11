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

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> qkv_causal_conv1d_silu(
    const ttnn::Tensor& input,
    const ttnn::Tensor& history,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const QkvCausalConv1dSiluProgramConfig& program_config,
    const ttnn::Tensor& actual_start,
    const ttnn::Tensor& predecessor_carry,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    uint32_t sequence_parallel_axis = 0,
    uint32_t input_column_offset = 0);

// As qkv_causal_conv1d_silu, with predecessor histories taken from history_table, the SP all-gather of
// convolution_history_rows, and the replacement three-row carry returned as a fourth output.
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> qkv_causal_conv1d_silu_with_carry(
    const ttnn::Tensor& input,
    const ttnn::Tensor& history,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    const QkvCausalConv1dSiluProgramConfig& program_config,
    const ttnn::Tensor& actual_start,
    const ttnn::Tensor& history_table,
    const std::optional<ttnn::Tensor>& actual_end = std::nullopt,
    uint32_t input_column_offset = 0,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    uint32_t sequence_parallel_axis = 0);

// Rows of a TILE-layout projection that the SP convolution exchanges, as a ROW_MAJOR [1, 6, width] tensor:
// the three rows preceding the next physical rank's segment, then the three rows ending at this rank's valid end
// (rows past a short end segment are unspecified). Columns start at column_offset.
ttnn::Tensor convolution_history_rows(
    const ttnn::Tensor& input,
    const ttnn::Tensor& actual_start,
    uint32_t width,
    uint32_t column_offset = 0,
    const std::optional<ttnn::Tensor>& actual_end = std::nullopt,
    uint32_t sequence_parallel_axis = 0);

}  // namespace ttnn::experimental::kda
