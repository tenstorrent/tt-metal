// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"

#include <tt-metalium/constants.hpp>

#include "llama_reduce_scatter.hpp"
#include "device/llama_reduce_scatter_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>

namespace ttnn::operations::experimental::ccl {
namespace detail {}  // namespace detail

ttnn::Tensor ExecuteLlamaReduceScatter::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& intermediate_packet_buffer,
    const int32_t dim,
    const GlobalSemaphore& cross_device_semaphore,
    const tt::tt_metal::SubDeviceId& subdevice_id,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    tt::tt_fabric::Topology topology) {
    const auto& mesh_view = mesh_device.get_view();
    const uint32_t ring_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();

    TT_FATAL(ring_devices > 1, "reduce_scatter async op will only work for ring_devices > 1, but has {}", ring_devices);

    return ttnn::prim::llama_reduce_scatter(
        input_tensor,
        intermediate_packet_buffer,
        dim,
        cross_device_semaphore,
        subdevice_id,
        cluster_axis,
        ring_devices,
        num_links,
        memory_config,
        topology);
}

}  // namespace ttnn::operations::experimental::ccl
