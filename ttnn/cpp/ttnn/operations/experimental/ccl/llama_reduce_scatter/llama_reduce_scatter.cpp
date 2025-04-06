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
    const global_semaphore::MultiDeviceGlobalSemaphore& cross_device_semaphore,
    const tt::tt_metal::SubDeviceId& subdevice_id,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    bool enable_persistent_fabric = true;

    const auto mesh_view = mesh_device.get_view();
    auto devices = input_tensor.get_workers();
    uint32_t ring_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    TT_FATAL(ring_devices > 1, "reduce_scatter async op will only work for ring_devices > 1, but has {}", ring_devices);

    ttnn::ccl::Topology ccl_topology = ttnn::ccl::Topology::Linear;

    std::vector<Tensor> output_tensors = {Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor}))};
    uint32_t input_tensor_index = 0;
    for (const auto& tensor : {input_tensor, intermediate_packet_buffer}) {
        auto buffers = tensor.buffers();
        auto first_address = buffers.front()->address();
        TT_FATAL(
            std::all_of(
                buffers.begin(),
                buffers.end(),
                [&first_address](const auto& buffer) {
                    return buffer != nullptr && buffer->address() == first_address;
                }),
            "Buffers must be lock-step allocated. Tensor on device id {} was allocated at different addresses "
            "from address {}",
            tensor.device()->id(),
            first_address);
    }

    std::vector<GlobalSemaphore> semaphores = cross_device_semaphore.global_semaphores;
    tt::tt_metal::operation::launch_op(
        [dim, semaphores, subdevice_id, cluster_axis, ring_devices, memory_config, mesh_view, num_links](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            auto input_tensor = input_tensors.at(0);
            auto intermediate_packet_buffer = input_tensors.at(1);
            uint32_t ring_index = 0;  // Initialize device index
            std::optional<IDevice*> forward_device = std::nullopt;
            std::optional<IDevice*> backward_device = std::nullopt;
            std::optional<GlobalSemaphore> semaphore = std::nullopt;
            const auto coordinate = mesh_view.find_device(input_tensor.device()->id());
            std::vector<IDevice*> devices = (cluster_axis == 0) ? mesh_view.get_devices_on_column(coordinate[1])
                                                                : mesh_view.get_devices_on_row(coordinate[0]);
            for (uint32_t i = 0; i < ring_devices; ++i) {
                if (devices.at(i) == input_tensor.device()) {
                    ring_index = i;
                    semaphore = semaphores.at(i);
                    if (i != 0) {
                        backward_device = devices.at(i - 1);
                    }
                    if (i != ring_devices - 1) {
                        forward_device = devices.at(i + 1);
                    }
                }
            }
            return {ttnn::prim::llama_reduce_scatter(
                input_tensor,
                intermediate_packet_buffer,
                dim,
                semaphore.value(),
                subdevice_id,
                ring_index,
                cluster_axis,
                forward_device,
                backward_device,
                ring_devices,
                num_links,
                memory_config)};
        },
        {input_tensor, intermediate_packet_buffer},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace ttnn::operations::experimental::ccl
