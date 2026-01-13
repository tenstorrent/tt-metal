// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_all_gather_async.hpp"
#include <utility>
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "device/ring_attention_all_gather_async_device_operation.hpp"

namespace ttnn::operations::experimental::ccl {

std::vector<ttnn::Tensor> ExecuteRingAttentionAllGatherAsync::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    std::vector<ttnn::Tensor>& persistent_output_buffer,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensors.at(0), topology, cluster_axis);
    return ttnn::prim::ring_attention_all_gather_async(
        input_tensors,
        persistent_output_buffer,
        dim,
        multi_device_global_semaphore,
        cluster_axis,
        mesh_device,
        topology_,
        num_links,
        memory_config,
        subdevice_id);
}

}  // namespace ttnn::operations::experimental::ccl
