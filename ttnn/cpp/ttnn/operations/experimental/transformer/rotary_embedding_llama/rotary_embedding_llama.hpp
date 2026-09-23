// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental {

ttnn::Tensor rotary_embedding_llama(
    const Tensor& input_tensor,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const Tensor& trans_mat,
    bool is_decode_mode = false,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    /// Fold the q/k RMS normalisation (x * rsqrt(mean_head_dim(x^2) + eps)) into the kernel; prefill interleaved only.
    std::optional<float> rms_norm_eps = std::nullopt);

}  // namespace ttnn::experimental
