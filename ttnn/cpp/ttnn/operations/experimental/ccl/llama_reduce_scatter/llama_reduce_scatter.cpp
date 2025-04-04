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
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    std::vector<Tensor> output_tensors = {Tensor(input_tensor.mesh_device())};

    const auto& mesh_view = mesh_device.get_view();
    const uint32_t ring_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    TT_FATAL(ring_devices > 1, "reduce_scatter async op will only work for ring_devices > 1, but has {}", ring_devices);

    tt::tt_metal::operation::launch_op(
        [dim, &cross_device_semaphore, subdevice_id, cluster_axis, memory_config, num_links, ring_devices](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            auto input_tensor = input_tensors.at(0);
            auto intermediate_packet_buffer = input_tensors.at(1);

            return {ttnn::prim::llama_reduce_scatter(
                input_tensor,
                intermediate_packet_buffer,
                dim,
                cross_device_semaphore,
                subdevice_id,
                cluster_axis,
                ring_devices,
                num_links,
                memory_config)};
        },
        {input_tensor, intermediate_packet_buffer},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace ttnn::operations::experimental::ccl
