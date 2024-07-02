// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/experimental//tt_dnn/op_library/update_cache/update_cache_op.hpp"

namespace ttnn {
namespace operations {
namespace kv_cache {

struct UpdateKVCache {
    static inline const std::array<TensorSchema, 2> input_tensor_schemas() {
        return {
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b},
                {ttnn::ROW_MAJOR_LAYOUT, ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false},
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b},
                {ttnn::ROW_MAJOR_LAYOUT, ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false},
        };
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const ttnn::Tensor& cache, const ttnn::Tensor& token, Args&&... args) {
        return std::forward_as_tuple(cache, token);
    }
};

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

constexpr auto fill_cache_for_user_ =
    ttnn::register_operation<ttnn::operations::kv_cache::ExecuteFillCache>("ttnn::fill_cache_for_user_");
constexpr auto update_cache_for_token_ =
    ttnn::register_operation<ttnn::operations::kv_cache::ExecuteUpdateCache>("ttnn::update_cache_for_token_");

}  // namespace ttnn
