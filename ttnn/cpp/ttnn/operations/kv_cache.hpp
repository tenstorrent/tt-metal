// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tt_dnn/op_library/update_cache/update_cache_op.hpp"

namespace ttnn {
namespace operations {
namespace kv_cache {

struct ExecuteFillCache {
    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& cache, const ttnn::Tensor& input, const uint32_t batch_index) {
        operation::run(
            tt::tt_metal::UpdateCache{batch_index, 0, 0, tt::tt_metal::UpdateCacheOpType::FILL},
            std::vector<ttnn::Tensor>{cache, input});
        return cache;
    }
};

struct ExecuteUpdateCache {
    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& cache,
        const ttnn::Tensor& input,
        const uint32_t update_index,
        const uint32_t batch_offset,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
        auto kernel_config_val = init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config);
        operation::run(
            tt::tt_metal::UpdateCache{
                0, update_index, batch_offset, tt::tt_metal::UpdateCacheOpType::UPDATE, kernel_config_val},
            std::vector<ttnn::Tensor>{cache, input});
        return cache;
    }
};

}  // namespace kv_cache
}  // namespace operations

namespace kv_cache {
constexpr auto fill_cache_for_user_ =
    ttnn::register_operation<ttnn::operations::kv_cache::ExecuteFillCache>("ttnn::kv_cache::fill_cache_for_user_");
constexpr auto update_cache_for_token_ =
    ttnn::register_operation<ttnn::operations::kv_cache::ExecuteUpdateCache>("ttnn::kv_cache::update_cache_for_token_");
}  // namespace kv_cache

}  // namespace ttnn
