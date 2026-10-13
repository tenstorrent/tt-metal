// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/transformer/sdpa/sdpa.hpp"

namespace ttnn::experimental::gemma4 {

std::tuple<ttnn::Tensor, ttnn::Tensor> rjsdpa(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_kv,
    ttnn::Tensor& persistent_output_buffer_kv,
    uint32_t head_dim_v,
    std::size_t logical_n,
    operations::transformer::SDPAProgramConfig program_config,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    std::optional<uint32_t> cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    CoreCoord ccl_core_grid_offset,
    bool is_balanced = false,
    std::optional<float> scale = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy = ttnn::ccl::CoreAllocationStrategy::ROW_MAJOR,
    std::optional<uint32_t> kv_cache_batch_idx = std::nullopt,
    std::optional<uint32_t> kv_actual_isl = std::nullopt,
    // Metadata / cache-fold contract: ring_joint_scaled_dot_product_attention docstring (sdpa_nanobind.cpp).
    // nullopt layers/idx resolve to 1/0.
    const std::optional<ttnn::Tensor>& slot_id = std::nullopt,
    const std::optional<ttnn::Tensor>& kv_actual_isl_tensor = std::nullopt,
    std::optional<uint32_t> kv_cache_num_layers = std::nullopt,
    std::optional<uint32_t> kv_cache_layer_idx = std::nullopt,
    std::optional<uint32_t> stripe_size = std::nullopt);

}  // namespace ttnn::experimental::gemma4
