// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_reduce_scatter_create_heads.hpp"
#include "device/llama_reduce_scatter_create_heads_device_op.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>

namespace ttnn::operations::experimental::ccl {
namespace detail {}  // namespace detail

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ExecuteLlamaReduceScatterCreateHeads::invoke(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& intermediate_packet_buffer,
    const int32_t dim,
    const GlobalSemaphore& cross_device_semaphore,
    const tt::tt_metal::SubDeviceId& subdevice_id,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    const uint32_t num_links,
    const uint32_t num_heads,
    const uint32_t num_kv_heads,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::MemoryConfig>& qkv_memory_config,
    bool use_noc1_only,
    bool use_optimal_ccl_for_llama) {
    const auto& mesh_view = mesh_device.get_view();
    uint32_t ring_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    TT_FATAL(ring_devices > 1, "reduce_scatter async op will only work for ring_devices > 1, but has {}", ring_devices);
    topology = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    uint32_t head_dim = input_tensor.padded_shape()[-1] / (num_heads + 2 * num_kv_heads);
    uint32_t slice_size = input_tensor.padded_shape()[-2] / ring_devices;
    auto output_tensors = ttnn::prim::llama_reduce_scatter_create_heads(
        input_tensor,
        intermediate_packet_buffer,
        dim,
        cross_device_semaphore,
        subdevice_id,
        cluster_axis,
        ring_devices,
        topology,
        num_links,
        num_heads,
        num_kv_heads,
        head_dim,
        slice_size,
        memory_config,
        qkv_memory_config,
        use_noc1_only,
        use_optimal_ccl_for_llama);
    return {output_tensors[0], output_tensors[1], output_tensors[2]};
}

}  // namespace ttnn::operations::experimental::ccl
