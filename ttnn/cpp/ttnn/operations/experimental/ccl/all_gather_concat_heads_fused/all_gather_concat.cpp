// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_concat.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/all_gather_concat_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "cpp/ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteAllGatherConcat::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& buffer_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_heads,
    const std::optional<uint32_t> num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    return ttnn::operations::experimental::ccl::all_gather_concat(
        input_tensor,
        buffer_tensor,
        dim,
        cluster_axis,
        mesh_device,
        multi_device_global_semaphore,
        num_heads,
        num_links,
        memory_config,
        topology,
        subdevice_id);
}

}  // namespace ttnn::operations::experimental::ccl
