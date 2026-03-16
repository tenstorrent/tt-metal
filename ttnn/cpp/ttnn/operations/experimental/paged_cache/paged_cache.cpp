// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/update_cache/paged_update_cache_device_operation.hpp"
#include "device/fused_update_cache/paged_fused_update_cache_device_operation.hpp"
#include "device/fill_cache/paged_fill_cache_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/experimental/paged_cache/paged_cache.hpp"

namespace ttnn::experimental {

ttnn::Tensor paged_update_cache(
    const Tensor& cache_tensor,
    const Tensor& input_tensor,
    const std::vector<uint32_t>& update_idxs,
    const std::optional<const Tensor>& update_idxs_tensor,
    std::optional<bool> share_cache,
    const std::optional<const Tensor>& page_table,
    uint32_t batch_offset,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const std::set<ttnn::MeshCoordinate>>& mesh_coords) {
    return ttnn::prim::paged_update_cache(
        cache_tensor,
        input_tensor,
        update_idxs,
        update_idxs_tensor,
        share_cache,
        page_table,
        batch_offset,
        compute_kernel_config,
        mesh_coords);
}

std::tuple<ttnn::Tensor, ttnn::Tensor> paged_fused_update_cache(
    const Tensor& cache_tensor1,
    const Tensor& input_tensor1,
    const Tensor& cache_tensor2,
    const Tensor& input_tensor2,
    const std::vector<uint32_t>& update_idxs,
    const std::optional<const Tensor>& update_idxs_tensor,
    std::optional<bool> share_cache,
    const std::optional<const Tensor>& page_table,
    uint32_t batch_offset,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const std::set<ttnn::MeshCoordinate>>& mesh_coords) {
    return ttnn::prim::paged_fused_update_cache(
        cache_tensor1,
        input_tensor1,
        cache_tensor2,
        input_tensor2,
        update_idxs,
        update_idxs_tensor,
        share_cache,
        page_table,
        batch_offset,
        compute_kernel_config,
        mesh_coords);
}

ttnn::Tensor paged_fill_cache(
    const Tensor& cache_tensor,
    const Tensor& input_tensor,
    const Tensor& page_table,
    const std::optional<const Tensor>& batch_idx_tensor,
    uint32_t batch_idx,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const std::set<ttnn::MeshCoordinate>>& mesh_coords) {
    // Note: compute_kernel_config is not used by fill_cache operation
    (void)compute_kernel_config;

    return ttnn::prim::paged_fill_cache(
        cache_tensor, input_tensor, page_table, batch_idx_tensor, batch_idx, mesh_coords);
}

}  // namespace ttnn::experimental
