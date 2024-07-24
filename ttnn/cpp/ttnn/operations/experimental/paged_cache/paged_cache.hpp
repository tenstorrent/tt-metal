// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/paged_cache_operation.hpp" // TODO: not right!
#include "ttnn/run_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/compute_kernel_config.hpp"
#include "ttnn/core.hpp"

namespace ttnn {
namespace operations::experimental::paged_cache {

struct PagedUpdateCacheOperation {
    static ttnn::Tensor execute_on_worker_thread(
        const Tensor& cache_tensor, const Tensor& input_tensor, const std::vector<uint32_t> update_idxs, const std::optional<const Tensor> update_idxs_tensor = std::nullopt, const std::optional<const Tensor> page_table = std::nullopt, const uint32_t batch_offset = 0, std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
        auto kernel_config_val = init_device_compute_kernel_config(input_tensor.device()->arch(), compute_kernel_config);
        operation::run(PagedUpdateCacheDeviceOperation{0, update_idxs, batch_offset, PagedUpdateCacheOpType::UPDATE, kernel_config_val}, {cache_tensor, input_tensor}, {update_idxs_tensor, page_table});

        return cache_tensor; // Updated cache tensor in-place
    }
};

struct PagedFillCacheOperation {
    static ttnn::Tensor execute_on_worker_thread(
        const Tensor& cache_tensor, const Tensor& input_tensor, const Tensor& page_table, const uint32_t batch_idx, std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
        auto kernel_config_val = init_device_compute_kernel_config(input_tensor.device()->arch(), compute_kernel_config);
        operation::run(PagedUpdateCacheDeviceOperation{batch_idx, {}, 0, PagedUpdateCacheOpType::FILL, kernel_config_val}, {cache_tensor, input_tensor, page_table}, {std::nullopt, std::nullopt});

        return cache_tensor; // Updated cache tensor in-place
    }
};

}  // namespace operations::experimental::paged_cache

namespace experimental::paged_cache {

constexpr auto paged_update_cache = ttnn::register_operation<ttnn::operations::experimental::paged_cache::PagedUpdateCacheOperation>(
    "ttnn::experimental::paged_cache::paged_update_cache");

constexpr auto paged_fill_cache = ttnn::register_operation<ttnn::operations::experimental::paged_cache::PagedFillCacheOperation>(
    "ttnn::experimental::paged_cache::paged_fill_cache");

}  // namespace experimental::paged_cache

}  // namespace ttnn
