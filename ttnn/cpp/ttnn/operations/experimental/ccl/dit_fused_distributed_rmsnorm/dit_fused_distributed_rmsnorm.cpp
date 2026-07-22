// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_fused_distributed_rmsnorm.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device.hpp"
#include "device/dit_fused_distributed_rmsnorm_device_operation.hpp"

using namespace tt::constants;

namespace ttnn::experimental {

namespace {

// Shared implementation for both public ops. norm_type selects RMS vs Welford LayerNorm;
// the fabric all-gather, weight/bias, RoPE and output plumbing are identical.
ttnn::Tensor dit_fused_distributed_norm_impl(
    const ttnn::Tensor& input_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const ttnn::ccl::Topology topology,
    const float epsilon,
    const uint32_t num_heads_per_device,
    const bool per_head_norm,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<const ttnn::Tensor>& transformation_mat,
    const std::optional<const ttnn::Tensor>& rope_cos,
    const std::optional<const ttnn::Tensor>& rope_sin,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const std::optional<size_t> num_preferred_links,
    const std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config,
    const DitFusedNormType norm_type,
    const std::optional<const ttnn::Tensor>& reciprocals) {
    return ttnn::prim::dit_fused_distributed_rmsnorm(
        input_tensor,
        cluster_axis,
        mesh_device,
        multi_device_global_semaphore,
        topology,
        epsilon,
        num_heads_per_device,
        per_head_norm,
        weight,
        bias,
        transformation_mat,
        rope_cos,
        rope_sin,
        dtype,
        persistent_output_buffer,
        num_preferred_links,
        subdevice_id,
        memory_config,
        compute_kernel_config,
        norm_type,
        reciprocals);
}

std::optional<ttnn::Tensor> dit_fused_distributed_norm_create_stats_buffer_impl(
    const ttnn::Tensor& input_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const uint32_t num_heads_per_device,
    const bool per_head_norm,
    const uint32_t num_links,
    // weight/RoPE are accepted for API symmetry with the op call (callers pass the same tensors),
    // but the stats-buffer geometry does not depend on them — see compute_sizing.
    [[maybe_unused]] const std::optional<const ttnn::Tensor>& weight,
    [[maybe_unused]] const std::optional<const ttnn::Tensor>& transformation_mat,
    [[maybe_unused]] const std::optional<const ttnn::Tensor>& rope_cos,
    [[maybe_unused]] const std::optional<const ttnn::Tensor>& rope_sin,
    const DitFusedNormType norm_type) {
    const auto& mesh_view = mesh_device.get_view();
    const std::size_t ring_size = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    auto arch = is_device_tensor(input_tensor) ? input_tensor.device()->arch() : ttnn::GetDefaultDevice()->arch();
    // fp32_dest_acc=true to MATCH the op's default — the chunk clamp's streaming
    // decision depends on the intermediate-CB tile size (fp32 vs bf16).
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, std::nullopt, tt::tt_metal::MathFidelity::HiFi4, true, true, false);

    auto params = ttnn::experimental::prim::DitFusedDistributedRmsnormParams(
        /*epsilon=*/0.0f,
        num_heads_per_device,
        per_head_norm,
        /*dtype=*/std::nullopt,
        input_tensor.memory_config(),
        cluster_axis,
        num_links,
        static_cast<uint32_t>(ring_size),
        ttnn::ccl::Topology::Ring,
        /*multi_device_global_semaphore=*/{},
        /*sub_device_id=*/std::nullopt,
        kernel_config_val,
        norm_type);

    // The stats-buffer geometry depends only on input shape / ring size / links / norm_type
    // (all in params + input_tensor), so no tensor_args are needed here.
    const auto sizing = ttnn::experimental::prim::compute_sizing(params, input_tensor);
    if (!sizing.use_mux) {
        return std::nullopt;
    }
    // Same spec compute_output_specs allocates, so the pre-allocated buffer matches the op exactly.
    const auto spec = ttnn::experimental::prim::make_stats_tensor_spec(sizing);
    return ttnn::create_device_tensor(spec, &const_cast<MeshDevice&>(mesh_device));
}

}  // namespace

ttnn::Tensor dit_fused_distributed_rmsnorm(
    const ttnn::Tensor& input_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const ttnn::ccl::Topology topology,
    const float epsilon,
    const uint32_t num_heads_per_device,
    const bool per_head_norm,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<const ttnn::Tensor>& transformation_mat,
    const std::optional<const ttnn::Tensor>& rope_cos,
    const std::optional<const ttnn::Tensor>& rope_sin,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const std::optional<size_t> num_preferred_links,
    const std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
    return dit_fused_distributed_norm_impl(
        input_tensor,
        cluster_axis,
        mesh_device,
        multi_device_global_semaphore,
        topology,
        epsilon,
        num_heads_per_device,
        per_head_norm,
        weight,
        bias,
        transformation_mat,
        rope_cos,
        rope_sin,
        dtype,
        persistent_output_buffer,
        num_preferred_links,
        subdevice_id,
        memory_config,
        compute_kernel_config,
        DitFusedNormType::RMS,
        /*reciprocals=*/std::nullopt);
}

ttnn::Tensor dit_fused_distributed_layernorm(
    const ttnn::Tensor& input_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const ttnn::ccl::Topology topology,
    const float epsilon,
    const uint32_t num_heads_per_device,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<const ttnn::Tensor>& transformation_mat,
    const std::optional<const ttnn::Tensor>& rope_cos,
    const std::optional<const ttnn::Tensor>& rope_sin,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const std::optional<size_t> num_preferred_links,
    const std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<const ttnn::Tensor>& reciprocals) {
    return dit_fused_distributed_norm_impl(
        input_tensor,
        cluster_axis,
        mesh_device,
        multi_device_global_semaphore,
        topology,
        epsilon,
        num_heads_per_device,
        /*per_head_norm=*/false,
        weight,
        bias,
        transformation_mat,
        rope_cos,
        rope_sin,
        dtype,
        persistent_output_buffer,
        num_preferred_links,
        subdevice_id,
        memory_config,
        compute_kernel_config,
        DitFusedNormType::LAYERNORM,
        reciprocals);
}

std::optional<ttnn::Tensor> dit_fused_distributed_rmsnorm_create_stats_buffer(
    const ttnn::Tensor& input_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const uint32_t num_heads_per_device,
    const bool per_head_norm,
    const uint32_t num_links,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& transformation_mat,
    const std::optional<const ttnn::Tensor>& rope_cos,
    const std::optional<const ttnn::Tensor>& rope_sin) {
    return dit_fused_distributed_norm_create_stats_buffer_impl(
        input_tensor,
        cluster_axis,
        mesh_device,
        num_heads_per_device,
        per_head_norm,
        num_links,
        weight,
        transformation_mat,
        rope_cos,
        rope_sin,
        DitFusedNormType::RMS);
}

std::optional<ttnn::Tensor> dit_fused_distributed_layernorm_create_stats_buffer(
    const ttnn::Tensor& input_tensor,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const uint32_t num_heads_per_device,
    const uint32_t num_links,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& transformation_mat,
    const std::optional<const ttnn::Tensor>& rope_cos,
    const std::optional<const ttnn::Tensor>& rope_sin) {
    return dit_fused_distributed_norm_create_stats_buffer_impl(
        input_tensor,
        cluster_axis,
        mesh_device,
        num_heads_per_device,
        /*per_head_norm=*/false,
        num_links,
        weight,
        transformation_mat,
        rope_cos,
        rope_sin,
        DitFusedNormType::LAYERNORM);
}

}  // namespace ttnn::experimental
