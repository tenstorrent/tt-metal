// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_device_operation.hpp"

namespace ttnn::transformer {

ttnn::Tensor scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& attn_mask = std::nullopt,
    bool is_causal = true,
    std::optional<float> scale = std::nullopt,
    std::optional<uint32_t> sliding_window_size = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& attention_sink = std::nullopt,
    const std::optional<ttnn::Tensor>& cu_window_seqlens = std::nullopt,
    /// Windowed mode only. Global row index of Q row 0, for a Q holding a contiguous slice of a longer
    /// sequence: Q and the output are indexed locally while cu_window_seqlens and K/V stay global, so this
    /// locates the slice among the windows. Must be a multiple of TILE_HEIGHT and satisfy offset+Sq <= Sk.
    uint32_t windowed_q_token_offset = 0,
    /// Windowed mode only. Per-device form of the offset above: a 1-element int32/uint32 ROW_MAJOR device
    /// tensor, read at runtime rather than baked into the program. Shard it on the sequence-parallel axis
    /// so every device runs the SAME program yet sees its own origin. Overrides the scalar when set.
    const std::optional<ttnn::Tensor>& windowed_q_token_offset_tensor = std::nullopt);

/// Chunked SDPA over paged K/V: one Q chunk per call, K/V in paged layout.
/// Two overloads: legacy (chunk_start_idx as int) or flexible (chunk_start_idx_tensor on device).

/// Legacy: chunk start index as scalar.
ttnn::Tensor chunked_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    int64_t chunk_start_idx,  // Must be a multiple of program_config.q_chunk_size
    std::optional<float> scale = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    // Geometry override for an HMA-shared paged cache. Q drives head_dim; supply this
    // call's view (block_size + num_kv_heads) when the cache was allocated for a different
    // layer. nullopt ⇒ cache shape.
    std::optional<operations::transformer::PagedCacheGeometryOverride> paged_cache_geometry = std::nullopt);

/// Flexible: chunk start index in device tensor [1] (int32). Read at runtime; use for trace.
ttnn::Tensor chunked_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& page_table_tensor,
    const ttnn::Tensor& chunk_start_idx_tensor,
    std::optional<float> scale = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<operations::transformer::PagedCacheGeometryOverride> paged_cache_geometry = std::nullopt);

std::tuple<ttnn::Tensor, ttnn::Tensor> joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& joint_tensor_q,
    const ttnn::Tensor& joint_tensor_k,
    const ttnn::Tensor& joint_tensor_v,
    const std::string& joint_strategy,
    operations::transformer::SDPAProgramConfig program_config,
    std::optional<float> scale = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ring_joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& joint_tensor_q,
    const std::optional<ttnn::Tensor>& joint_tensor_k,
    const std::optional<ttnn::Tensor>& joint_tensor_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
    const std::string& joint_strategy,
    std::size_t logical_n,
    std::size_t logical_l,
    operations::transformer::SDPAProgramConfig program_config,
    int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    CoreCoord ccl_core_grid_offset,
    bool is_causal = false,
    bool is_balanced = false,
    bool is_cross = false,
    std::optional<float> scale = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy = ttnn::ccl::CoreAllocationStrategy::ROW_MAJOR,
    std::optional<uint32_t> kv_cache_batch_idx = std::nullopt,
    std::optional<uint32_t> kv_actual_isl = std::nullopt,
    const std::optional<ttnn::Tensor>& attention_sink = std::nullopt,
    std::optional<uint32_t> sliding_window_size = std::nullopt,
    const std::optional<ttnn::Tensor>& persistent_output_buffer_joint_k = std::nullopt,
    const std::optional<ttnn::Tensor>& persistent_output_buffer_joint_v = std::nullopt,
    std::optional<uint32_t> kv_cache_num_layers = std::nullopt,
    std::optional<uint32_t> kv_cache_layer_idx = std::nullopt,
    const std::optional<ttnn::Tensor>& page_bundle_indices = std::nullopt,
    uint32_t kv_cache_page_size = 32);

std::tuple<ttnn::Tensor, ttnn::Tensor> ring_mla(
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
    // Trace-safe metadata path: when set (both together), the per-chunk scalars (kv_cache_batch_idx /
    // kv_actual_isl / logical_n) are read on-device from these two 1-element uint32 DRAM tensors instead
    // of being baked into the program, so one captured ttnn trace replays across chunks. slot_id holds
    // the cache-user slot (was metadata[0]); kv_actual_isl_tensor holds the prior valid global KV length
    // (was metadata[1]).
    const std::optional<ttnn::Tensor>& slot_id = std::nullopt,
    const std::optional<ttnn::Tensor>& kv_actual_isl_tensor = std::nullopt,
    // Layer geometry for metadata cache-slot selection and paged physical bundles. The metadata
    // readers compute slot_id[0] * kv_cache_num_layers + kv_cache_layer_idx (mirrors
    // update_padded_kv_cache). Defaults 1/0 preserve the existing single-layer behavior.
    std::optional<uint32_t> kv_cache_num_layers = std::nullopt,
    std::optional<uint32_t> kv_cache_layer_idx = std::nullopt,
    // Paged MLA cache: ordered logical-local page list. The physical cache is
    // [num_bundles * num_layers * num_heads, 1, kv_cache_page_size, head_dim], with
    // flat_page = (bundle * num_layers + layer) * num_heads + head. Every uint16
    // page_bundle_indices value must be smaller than num_bundles.
    const std::optional<ttnn::Tensor>& page_bundle_indices = std::nullopt,
    uint32_t kv_cache_page_size = 32);

struct ExecuteExpRingJointAttention {
    static std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& input_tensor_v,
        const std::optional<ttnn::Tensor>& joint_tensor_q,
        const std::optional<ttnn::Tensor>& joint_tensor_k,
        const std::optional<ttnn::Tensor>& joint_tensor_v,
        ttnn::Tensor& persistent_output_buffer_k,
        ttnn::Tensor& persistent_output_buffer_v,
        const std::string& joint_strategy,
        std::size_t logical_n,
        operations::transformer::SDPAProgramConfig program_config,
        int32_t dim,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        uint32_t num_links,
        uint32_t cluster_axis,
        const MeshDevice& mesh_device,
        ttnn::ccl::Topology topology,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
        std::optional<float> scale = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        uint32_t num_workers_per_link = 1,
        uint32_t num_buffers_per_channel = 8);
};

ttnn::Tensor flash_mla_prefill(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    uint32_t head_dim_v,
    const std::optional<ttnn::Tensor>& input_tensor_v = std::nullopt,
    const std::optional<ttnn::Tensor>& attn_mask = std::nullopt,
    bool is_causal = true,
    std::optional<float> scale = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

ttnn::Tensor chunked_flash_mla_prefill(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    uint32_t head_dim_v,
    const ttnn::Tensor& page_table_tensor,
    int64_t chunk_start_idx,
    std::optional<float> scale = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

ttnn::Tensor ring_distributed_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    uint32_t ring_size,
    std::optional<uint32_t> ring_id =
        std::nullopt,  // Optional: if provided, uses this value; if nullopt, infers from device coordinate
    std::optional<float> scale = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<operations::transformer::SDPAProgramConfig>& program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& page_table = std::nullopt,
    std::optional<int64_t> chunk_start_idx = std::nullopt);

}  // namespace ttnn::transformer
