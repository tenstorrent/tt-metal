// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/gemma4/sliding_ring_sdpa/sliding_ring_sdpa.hpp"

#include <array>
#include <utility>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/gemma4/sliding_ring_sdpa/device/sliding_ring_sdpa_device_operation.hpp"

namespace ttnn::experimental::gemma4 {

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
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy,
    std::optional<uint32_t> kv_cache_batch_idx,
    std::optional<uint32_t> kv_actual_isl,
    const std::optional<ttnn::Tensor>& slot_id,
    const std::optional<ttnn::Tensor>& kv_actual_isl_tensor,
    std::optional<uint32_t> kv_cache_num_layers,
    std::optional<uint32_t> kv_cache_layer_idx) {
    // The ring size comes from mesh_device, but halo routing and dispatch follow Q's mesh: every tensor the op reads
    // or writes must live on it. The halo exchange writes the two halo buffers (last) while attention reads Q, K and
    // V, so each halo buffer needs storage of its own; shared with the other, V would land where K should be.
    const std::array<const Tensor*, 5> tensors = {
        &input_tensor_q, &input_tensor_k, &input_tensor_v, &persistent_output_buffer_k, &persistent_output_buffer_v};
    for (size_t i = 0; i < tensors.size(); ++i) {
        TT_FATAL(
            tensors[i]->device() == &mesh_device,
            "sliding_ring_sdpa: Q, K, V and the halo buffers must all be on mesh_device");
        for (size_t halo = 3; halo < tensors.size(); ++halo) {
            TT_FATAL(
                i == halo || &tensors[i]->mesh_buffer() != &tensors[halo]->mesh_buffer(),
                "sliding_ring_sdpa: each halo buffer must have its own storage, not shared with Q, K, V or the other "
                "halo buffer");
        }
    }
    // The shared op's call with Gemma4's fixed choices: causal, unbalanced, no joint tensors ("rear" is inert
    // without them), no attention sink, no circular cache, host logical lengths.
    auto output_tensors = ttnn::prim::gemma4_sliding_ring_sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        std::nullopt,  // joint_tensor_q
        std::nullopt,  // joint_tensor_k
        std::nullopt,  // joint_tensor_v
        persistent_output_buffer_k,
        persistent_output_buffer_v,
        std::nullopt,  // persistent_output_buffer_joint_k
        std::nullopt,  // persistent_output_buffer_joint_v
        "rear",
        logical_n,
        0,  // logical_l
        std::move(program_config),
        dim,
        multi_device_global_semaphore,
        num_links,
        cluster_axis,
        mesh_device,
        ttnn::ccl::convert_2d_to_1d_topology(topology),
        ccl_core_grid_offset,
        subdevice_id,
        /*is_causal=*/true,
        /*is_balanced=*/false,
        /*is_cross=*/false,
        scale,
        compute_kernel_config,
        core_allocation_strategy,
        kv_cache_batch_idx,
        kv_actual_isl,
        std::nullopt,  // latent_v_head_dim
        std::nullopt,  // attention_sink
        slot_id,
        kv_actual_isl_tensor,
        kv_cache_num_layers.value_or(1),
        kv_cache_layer_idx.value_or(0),
        sliding_window_size);
    return output_tensors[ttnn::prim::GEMMA4_SLIDING_RING_SDPA_OUTPUT_IDX];
}

}  // namespace ttnn::experimental::gemma4
