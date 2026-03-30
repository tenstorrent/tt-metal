// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental {

std::tuple<Tensor, Tensor> rotary_embedding_llama_fused_qk(
    const Tensor& q_input_tensor,
    const Tensor& k_input_tensor,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const Tensor& trans_mat,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental
