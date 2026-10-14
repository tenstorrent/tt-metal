// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rjsdpa.hpp"

#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.hpp"

namespace ttnn::experimental::gemma4 {

std::tuple<ttnn::Tensor, ttnn::Tensor> rjsdpa(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_kv,
    ttnn::Tensor& persistent_output_buffer_kv,
    const uint32_t head_dim_v,
    std::size_t logical_n,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<uint32_t> cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const CoreCoord ccl_core_grid_offset,
    bool is_balanced,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy,
    std::optional<uint32_t> kv_cache_batch_idx,
    std::optional<uint32_t> kv_actual_isl,
    const std::optional<ttnn::Tensor>& slot_id,
    const std::optional<ttnn::Tensor>& kv_actual_isl_tensor,
    std::optional<uint32_t> kv_cache_num_layers,
    std::optional<uint32_t> kv_cache_layer_idx,
    std::optional<uint32_t> stripe_size,
    const std::optional<ttnn::Tensor>& input_tensor_v,
    const std::optional<ttnn::Tensor>& persistent_output_buffer_v,
    std::optional<uint32_t> sliding_window_size,
    const std::string& communication) {
    TT_FATAL(
        communication == "all_gather" || communication == "halo", "rjsdpa communication must be all_gather or halo");
    const bool striped_halo = communication == "halo";
    TT_FATAL(
        !striped_halo || (stripe_size.has_value() && sliding_window_size.value_or(0) > 1 && input_tensor_v.has_value()),
        "rjsdpa halo requires explicit stripes, separate K/V, and sliding_window_size > 1");
    TT_FATAL(
        !striped_halo || (cluster_axis.has_value() && topology == ttnn::ccl::Topology::Linear),
        "Striped halo requires a CP axis and Linear topology");
    TT_FATAL(
        input_tensor_v.has_value() == persistent_output_buffer_v.has_value(),
        "rjsdpa separate V requires both input_tensor_v and persistent_output_buffer_v");
    TT_FATAL(
        !input_tensor_v.has_value() || input_tensor_v->logical_shape()[3] == head_dim_v,
        "rjsdpa head_dim_v must match the separate V tensor width");
    if (stripe_size.has_value()) {
        TT_FATAL(
            *stripe_size > 0 && *stripe_size % tt::constants::TILE_HEIGHT == 0,
            "rjsdpa stripe_size must be a positive multiple of 32 tokens");
        // Without an explicit prefix, preserve ring_mla's latest-Q-window convention.
        // Tensor metadata must remain device-resident for trace replay.
        if (!kv_actual_isl.has_value() && !kv_actual_isl_tensor.has_value()) {
            TT_FATAL(
                !cluster_axis.has_value() || *cluster_axis < mesh_device.shape().dims(),
                "rjsdpa cluster_axis must be within the mesh dimensions");
            const uint32_t ranks =
                cluster_axis.has_value() ? mesh_device.shape()[*cluster_axis] : mesh_device.shape().mesh_size();
            const uint64_t query_capacity = static_cast<uint64_t>(input_tensor_q.logical_shape()[2]) * ranks;
            kv_actual_isl = logical_n > query_capacity ? logical_n - query_capacity : 0;
        }
    }
    auto output_tensors = ttnn::prim::ring_joint_scaled_dot_product_attention(
        input_tensor_q,
        input_tensor_kv,
        input_tensor_v,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        persistent_output_buffer_kv,
        persistent_output_buffer_v,
        std::nullopt,  // persistent_output_buffer_joint_k
        std::nullopt,  // persistent_output_buffer_joint_v
        "rear",
        logical_n,
        /*logical_l=*/static_cast<std::size_t>(0),
        std::move(program_config),
        dim,
        multi_device_global_semaphore,
        num_links,
        cluster_axis,
        mesh_device,
        topology,
        ccl_core_grid_offset,
        subdevice_id,
        /*is_causal=*/true,
        is_balanced,
        /*is_cross=*/false,
        scale,
        compute_kernel_config,
        core_allocation_strategy,
        kv_cache_batch_idx,
        kv_actual_isl,
        input_tensor_v.has_value() ? std::nullopt : std::optional<uint32_t>(head_dim_v),
        std::nullopt,  // attention_sink
        slot_id,
        kv_actual_isl_tensor,
        kv_cache_num_layers.value_or(1),
        kv_cache_layer_idx.value_or(0),
        sliding_window_size,
        false,         // circular_kv_cache
        std::nullopt,  // logical_n_tensor
        std::nullopt,  // logical_l_tensor
        stripe_size.value_or(0),
        striped_halo);
    return {
        output_tensors[ttnn::prim::RING_JOINT_SDPA_OUTPUT_IDX],
        output_tensors[ttnn::prim::RING_JOINT_SDPA_STATS_OUTPUT_IDX]};
}

}  // namespace ttnn::experimental::gemma4
