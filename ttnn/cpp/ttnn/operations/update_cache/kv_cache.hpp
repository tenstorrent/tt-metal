// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "tt_dnn/op_library/compute_kernel_config.hpp"

namespace ttnn {
namespace operations {
namespace kv_cache {

struct ExecuteFillCache {
    static ttnn::Tensor operator()(
        const ttnn::Tensor& cache,
        const ttnn::Tensor& input,
        const uint32_t batch_index);
};

struct ExecuteUpdateCache {
    static ttnn::Tensor operator()(
        const ttnn::Tensor& cache,
        const ttnn::Tensor& input,
        const uint32_t update_index,
        const uint32_t batch_offset,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ExecuteStructUpdateCache {
    static ttnn::Tensor operator()(
        const ttnn::Tensor& cache,
        const ttnn::Tensor& input,
        const uint32_t update_idx,
        const uint32_t batch_offset,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ExecuteStructFillCache {
    static ttnn::Tensor operator()(
        const ttnn::Tensor& cache_tensor,
        const ttnn::Tensor& input_tensor,
        const uint32_t batch_idx);
};

}  // namespace kv_cache
}  // namespace operations

namespace kv_cache {
constexpr auto fill_cache_for_user_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::kv_cache::fill_cache_for_user_",
    ttnn::operations::kv_cache::ExecuteFillCache>();
constexpr auto update_cache_for_token_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::kv_cache::update_cache_for_token_",
    ttnn::operations::kv_cache::ExecuteUpdateCache>();

namespace kv_cache_utils {
    constexpr auto update_cache = ttnn::register_operation<"ttnn::update_cache", ttnn::operations::kv_cache::ExecuteStructUpdateCache>();
    constexpr auto fill_cache = ttnn::register_operation<"ttnn::fill_cache", ttnn::operations::kv_cache::ExecuteStructFillCache>();
}

}  // namespace kv_cache

}  // namespace ttnn
