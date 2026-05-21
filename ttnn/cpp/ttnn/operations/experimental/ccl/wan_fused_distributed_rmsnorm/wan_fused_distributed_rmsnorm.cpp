// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "wan_fused_distributed_rmsnorm.hpp"

#include "device/wan_fused_distributed_rmsnorm_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/all_gather_async.hpp"
#include "ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/rmsnorm_post_all_gather.hpp"
#include "ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/rmsnorm_pre_all_gather.hpp"

namespace ttnn::experimental {

ttnn::Tensor wan_fused_distributed_rmsnorm(
    const ttnn::Tensor& input_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const ttnn::ccl::Topology topology,
    const float epsilon,
    const uint32_t num_heads_per_device,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& transformation_mat,
    const std::optional<const ttnn::Tensor>& rope_cos,
    const std::optional<const ttnn::Tensor>& rope_sin,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const std::optional<size_t> num_preferred_links,
    const std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config,
    const bool use_device_op) {
    if (use_device_op) {
        // Dispatch to the new fused device op. Currently supports TP=1 only.
        return ttnn::prim::wan_fused_distributed_rmsnorm(
            input_tensor,
            cluster_axis,
            mesh_device,
            multi_device_global_semaphore,
            topology,
            epsilon,
            num_heads_per_device,
            weight,
            transformation_mat,
            rope_cos,
            rope_sin,
            dtype,
            persistent_output_buffer,
            num_preferred_links,
            subdevice_id,
            memory_config,
            compute_kernel_config);
    }

    // Stage 1: per-row partial stats (sum of squares) in fp32.
    ttnn::Tensor stats = ttnn::experimental::wan_fused_rmsnorm_pre_allgather(
        input_tensor,
        /*dtype=*/DataType::FLOAT32,
        compute_kernel_config,
        /*memory_config=*/std::nullopt);

    // Stage 2: all-gather the stats across the TP cluster axis.
    // Skip when the cluster axis has a single device — stats are already complete.
    const auto& mesh_view = mesh_device.get_view();
    const std::size_t cluster_size = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    ttnn::Tensor gathered_stats = stats;
    if (cluster_size > 1) {
        gathered_stats = ttnn::experimental::all_gather_async(
            stats,
            /*dim=*/3,
            cluster_axis,
            mesh_device,
            topology,
            multi_device_global_semaphore,
            persistent_output_buffer,
            /*memory_config=*/std::nullopt,
            num_preferred_links,
            subdevice_id);
    }

    // Stage 3: finalize normalization, optionally splitting heads, applying RoPE, and casting dtype.
    return ttnn::experimental::wan_fused_rmsnorm_post_allgather(
        input_tensor,
        gathered_stats,
        epsilon,
        num_heads_per_device,
        weight,
        transformation_mat,
        rope_cos,
        rope_sin,
        memory_config,
        compute_kernel_config,
        dtype);
}

}  // namespace ttnn::experimental
