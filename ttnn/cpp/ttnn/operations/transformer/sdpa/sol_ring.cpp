// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/sol_ring.hpp"
#include "ttnn/operations/transformer/sdpa/device/sol_ring_device_operation.hpp"
#include <tt-metalium/hal.hpp>

namespace ttnn::transformer {

namespace {
// A joint tensor with no rows is how callers spell "absent"; the op wants nullopt.
std::optional<ttnn::Tensor> drop_if_empty(const std::optional<ttnn::Tensor>& t) {
    if (t.has_value() && t->logical_shape()[-2] == 0) {
        return std::nullopt;
    }
    return t;
}
}  // namespace

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> sol_ring(
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
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<float> scale,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const uint32_t num_workers_per_link,
    const uint32_t num_buffers_per_channel,
    std::optional<ttnn::Tensor> chunk_mask,
    uint32_t chunk_keep_every) {
    // Normalize empty joints to nullopt (see drop_if_empty).
    const std::optional<ttnn::Tensor> joint_q = drop_if_empty(joint_tensor_q);
    const std::optional<ttnn::Tensor> joint_k = drop_if_empty(joint_tensor_k);
    const std::optional<ttnn::Tensor> joint_v = drop_if_empty(joint_tensor_v);

    auto output_tensors = ttnn::prim::sol_ring(
        input_tensor_q,
        input_tensor_k,  // AllGather input
        input_tensor_v,  // AllGather input
        joint_q,
        joint_k,
        joint_v,
        persistent_output_buffer_k,  // AllGather output / RingAttention input
        persistent_output_buffer_v,  // AllGather output / RingAttention input
        joint_strategy,
        logical_n,
        std::move(program_config),
        dim,
        multi_device_global_semaphore,
        num_links,
        cluster_axis,
        mesh_device,
        topology,
        subdevice_id,
        scale,
        compute_kernel_config,
        num_workers_per_link,
        num_buffers_per_channel,
        std::move(chunk_mask),
        chunk_keep_every);
    return {
        output_tensors[prim::SOL_RING_OUTPUT_IDX],
        output_tensors[prim::SOL_RING_JOINT_OUTPUT_IDX],
        output_tensors[prim::SOL_RING_STATS_OUTPUT_IDX]};
}

}  // namespace ttnn::transformer
