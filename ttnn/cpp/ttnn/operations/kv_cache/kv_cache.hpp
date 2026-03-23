// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {

ttnn::Tensor fill_cache_for_user_(const ttnn::Tensor& cache, const ttnn::Tensor& input, uint32_t batch_index);

ttnn::Tensor update_cache_for_token_(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    uint32_t update_index,
    uint32_t batch_offset,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

ttnn::Tensor update_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    uint32_t update_idx,
    uint32_t batch_offset = 0,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

ttnn::Tensor fill_cache(const ttnn::Tensor& cache_tensor, const ttnn::Tensor& input_tensor, uint32_t batch_idx);

}  // namespace ttnn
