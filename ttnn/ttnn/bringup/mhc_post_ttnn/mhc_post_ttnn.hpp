// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::mhc_post_ttnn {

// HiFi4, fp32 DEST, approx off (mhc_post.py default_compute_kernel_config()).
tt::tt_metal::ComputeConfigDescriptor default_compute_kernel_config();

// ttnn.bringup.mhc_post: X'[t, j*C:(j+1)*C] = post[t, j] * F[t, :] + sum_i comb[t, i*n + j] * X[t, i*C:(i+1)*C].
// The host-side checks of mhc_post.py (SUPPORTED axes, then the shape contract), in the same order and
// with the same exception types, then one device program. comb_transposed=false applies comb as stored:
// X'[t, j*C:(j+1)*C] = post[t, j] * F[t, :] + sum_i comb[t, j*n + i] * X[t, i*C:(i+1)*C].
Tensor mhc_post(
    const Tensor& input_tensor,
    const Tensor& residual,
    const Tensor& post,
    const Tensor& comb,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config = std::nullopt,
    bool comb_transposed = true,
    int64_t post_col = 0,
    int64_t comb_col = 0,
    const std::optional<Tensor>& output_tensor = std::nullopt);

}  // namespace ttnn::operations::bringup::mhc_post_ttnn

namespace ttnn::operations::bringup::mhc_post_ttnn {

// ttnn.bringup.mhc_post(..., stats_weight=W) (deepseek_v41_flash_d_p P.14): the mix above AND, from the same pass,
// the next hc step's partial row stats (..., T, 32) fp32 = [X' @ W | ...] with column n (n + 2) = sum_c X'^2
// (= mhc_pre_xing_pack(X' @ W, X')). W (..., n*C, 32) fp32 TILE (leading dims 1). Needs fp32 F / X, n <= 4.
std::tuple<Tensor, Tensor> mhc_post_with_stats(
    const Tensor& input_tensor,
    const Tensor& residual,
    const Tensor& post,
    const Tensor& comb,
    const Tensor& stats_weight,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config = std::nullopt,
    bool comb_transposed = true,
    int64_t post_col = 0,
    int64_t comb_col = 0,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const std::optional<Tensor>& stats_output = std::nullopt,
    const std::optional<Tensor>& stats_scratch = std::nullopt);

// The Python entry: X' without stats_weight (the op as it was), (X', stats) with it.
std::variant<Tensor, std::tuple<Tensor, Tensor>> mhc_post_py(
    const Tensor& input_tensor,
    const Tensor& residual,
    const Tensor& post,
    const Tensor& comb,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    bool comb_transposed,
    const std::optional<Tensor>& stats_weight,
    int64_t post_col = 0,
    int64_t comb_col = 0,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const std::optional<Tensor>& stats_output = std::nullopt,
    const std::optional<Tensor>& stats_scratch = std::nullopt);

}  // namespace ttnn::operations::bringup::mhc_post_ttnn
