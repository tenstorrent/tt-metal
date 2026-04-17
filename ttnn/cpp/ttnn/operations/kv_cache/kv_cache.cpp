// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kv_cache.hpp"
#include "ttnn/operations/kv_cache/device/update_cache_device_operation.hpp"
#include "ttnn/operations/kv_cache/device/zero_cache_range_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn {

ttnn::Tensor update_cache_for_token_(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    const uint32_t update_index,
    const uint32_t batch_offset,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto kernel_config_val = init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config);
    ttnn::prim::update_cache(
        cache, input, 0, update_index, batch_offset, ttnn::prim::UpdateCacheOpType::UPDATE, kernel_config_val);
    return cache;
}

ttnn::Tensor fill_cache_for_user_(
    const ttnn::Tensor& cache, const ttnn::Tensor& input, const uint32_t batch_index) {
    ttnn::prim::update_cache(cache, input, batch_index, 0, 0, ttnn::prim::UpdateCacheOpType::FILL);
    return cache;
}

ttnn::Tensor update_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    const uint32_t update_idx,
    const uint32_t batch_offset,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto kernel_config_val = init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config);
    ttnn::prim::update_cache(
        cache, input, 0, update_idx, batch_offset, ttnn::prim::UpdateCacheOpType::UPDATE, kernel_config_val);
    return cache;
}

ttnn::Tensor fill_cache(
    const ttnn::Tensor& cache_tensor, const ttnn::Tensor& input_tensor, const uint32_t batch_idx) {
    ttnn::prim::update_cache(cache_tensor, input_tensor, batch_idx, 0, 0, ttnn::prim::UpdateCacheOpType::FILL);
    return cache_tensor;
}

ttnn::Tensor zero_cache_range(const ttnn::Tensor& cache, const uint32_t start_token, const uint32_t end_token) {
    using namespace tt::constants;
    uint32_t Wt = cache.padded_shape()[-1] / TILE_WIDTH;
    // Round start_token down to tile boundary, end_token up to tile boundary
    uint32_t start_page = (start_token / TILE_HEIGHT) * Wt;
    uint32_t end_page = ((end_token + TILE_HEIGHT - 1) / TILE_HEIGHT) * Wt;
    ttnn::prim::zero_cache_range(cache, start_page, end_page);
    return cache;
}

}  // namespace ttnn
