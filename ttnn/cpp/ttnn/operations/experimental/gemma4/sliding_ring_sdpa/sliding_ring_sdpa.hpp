// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::gemma4 {

// Gemma4's sliding-window layers: causal chunked ring SDPA over separate K and V caches with the fused
// neighbor-halo exchange. Returns the attention output for Q's shape.
ttnn::Tensor sliding_ring_sdpa(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
    std::size_t logical_n,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    uint32_t sliding_window_size,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    CoreCoord ccl_core_grid_offset,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt,
    std::optional<float> scale = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy = ttnn::ccl::CoreAllocationStrategy::ROW_MAJOR,
    std::optional<uint32_t> kv_cache_batch_idx = std::nullopt,
    std::optional<uint32_t> kv_actual_isl = std::nullopt,
    const std::optional<ttnn::Tensor>& slot_id = std::nullopt,
    const std::optional<ttnn::Tensor>& kv_actual_isl_tensor = std::nullopt,
    std::optional<uint32_t> kv_cache_num_layers = std::nullopt,
    std::optional<uint32_t> kv_cache_layer_idx = std::nullopt);

}  // namespace ttnn::experimental::gemma4
