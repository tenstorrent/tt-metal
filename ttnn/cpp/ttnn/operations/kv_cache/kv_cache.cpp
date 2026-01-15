// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kv_cache.hpp"
#include "ttnn/operations/kv_cache/device/update_cache_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::kv_cache {

ttnn::Tensor ExecuteUpdateCache::invoke(
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

ttnn::Tensor ExecuteFillCache::invoke(
    const ttnn::Tensor& cache, const ttnn::Tensor& input, const uint32_t batch_index) {
    ttnn::prim::update_cache(cache, input, batch_index, 0, 0, ttnn::prim::UpdateCacheOpType::FILL);
    return cache;
}

ttnn::Tensor UpdateCacheOperation::invoke(
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

ttnn::Tensor FillCacheOperation::invoke(
    const ttnn::Tensor& cache_tensor, const ttnn::Tensor& input_tensor, const uint32_t batch_idx) {
    ttnn::prim::update_cache(cache_tensor, input_tensor, batch_idx, 0, 0, ttnn::prim::UpdateCacheOpType::FILL);
    return cache_tensor;
}

}  // namespace ttnn::operations::kv_cache
