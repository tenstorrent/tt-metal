// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::mhc_pre_ttnn {

// HiFi4, fp32 DEST, approx off (mhc_pre.py default_compute_kernel_config()).
tt::tt_metal::ComputeConfigDescriptor default_compute_kernel_config();

// ttnn.bringup.mhc_pre: per token row x of X (..., T, n*C): r = rsqrt(mean(x^2) + norm_eps), mixes = (x @ W) * r,
// pre / post / comb (Sinkhorn) coefficients and y = sum_i pre[i] * x[i*C:(i+1)*C]; returns (y, post, comb). The
// host-side checks of mhc_pre.py (SUPPORTED axes, then the structural contract), in the same order and with the same
// exception types, then one device program.
std::tuple<Tensor, Tensor, Tensor> mhc_pre(
    const Tensor& input_tensor,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    const std::vector<double>& scale,
    int64_t sinkhorn_iters = 20,
    double eps = 1e-6,
    double norm_eps = 1e-6,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn
