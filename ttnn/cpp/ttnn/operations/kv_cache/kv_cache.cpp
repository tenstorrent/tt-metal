// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kv_cache.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/kv_cache/device/update_cache_op.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"


namespace ttnn::operations::kv_cache {

ttnn::Tensor ExecuteUpdateCache::invoke(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    const uint32_t update_index,
    const uint32_t batch_offset,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
        auto kernel_config_val = init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config);
        operation::run(
            UpdateCache{
                0, update_index, batch_offset, UpdateCacheOpType::UPDATE, kernel_config_val},
            std::vector<ttnn::Tensor>{cache, input});
        return cache;
}

ttnn::Tensor ExecuteFillCache::invoke(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    const uint32_t batch_index) {
    operation::run(
            UpdateCache{batch_index, 0, 0, UpdateCacheOpType::FILL},
            std::vector<ttnn::Tensor>{cache, input});
        return cache;
}

ttnn::Tensor UpdateCacheOperation::invoke(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    const uint32_t update_idx,
    const uint32_t batch_offset,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    return update_cache_impl(cache, input, update_idx, batch_offset, compute_kernel_config);
}

ttnn::Tensor FillCacheOperation::invoke(
    const ttnn::Tensor& cache_tensor,
    const ttnn::Tensor& input_tensor,
    const uint32_t batch_idx) {
    return fill_cache_impl(cache_tensor, input_tensor, batch_idx);
}

}  // namespace ttnn::operations::kv_cache
