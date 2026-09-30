// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

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
    bool comb_transposed = true);

}  // namespace ttnn::operations::bringup::mhc_post_ttnn
