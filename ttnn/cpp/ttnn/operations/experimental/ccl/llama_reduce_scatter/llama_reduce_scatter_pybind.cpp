// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_reduce_scatter_pybind.hpp"

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "llama_reduce_scatter.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/fabric_edm_types.hpp>


namespace ttnn::operations::experimental::ccl {

void py_bind_llama_reduce_scatter(py::module& module) {
    auto doc =
        R"doc(llama_reduce_scatter(input_tensor: ttnn.Tensor, dims: List[int], memory_config: Optional[MemoryConfig] = std::nullopt, queue_id: int = 0) -> ttnn.Tensor

            Reduce_scatter after FF1/3 for Llama70B.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                intermediate_packet_buffer (ttnn.Tensor): the intermediate packet buffer tensor.
                dim (number): the reduce dimension
                cross_device_semaphore (ttnn.GlobalSemaphore): the cross device semaphore.
                subdevice_id (ttnn.SubDeviceId): the subdevice id.
                cluster_axis (number): the cluster axis.
                mesh_device (ttnn.MeshDevice): the mesh device.
                num_links (number, optional): the number of links. Defaults to `3`.

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                queue_id (int, optional): command queue id. Defaults to `0`.

           Returns:
               ttnn.Tensor: the output tensor.

            Example:

                >>> tensor = ttnn.experimental.llama_reduce_scatter(
                                tt_input_tensors_list[i],
                                tt_intermediate_tensors_list[i],
                                dim,
                                ccl_semaphore_handles[i],
                                worker_sub_device_id,
                                cluster_axis=1,
                                mesh_device=mesh_device,
                                num_links=num_links,
                                memory_config=output_mem_config))doc";

    using OperationType = decltype(ttnn::experimental::llama_reduce_scatter);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::llama_reduce_scatter,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               ttnn::Tensor& intermediate_packet_buffer,
               uint32_t dim,
               const GlobalSemaphore& cross_device_semaphore,
               const tt::tt_metal::SubDeviceId& subdevice_id,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               tt::tt_fabric::Topology topology,
               QueueId queue_id) {
                return self(
                    queue_id,
                    input_tensor,
                    intermediate_packet_buffer,
                    dim,
                    cross_device_semaphore,
                    subdevice_id,
                    cluster_axis,
                    mesh_device,
                    num_links,
                    memory_config,
                    topology);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("intermediate_packet_buffer").noconvert(),
            py::arg("dim"),
            py::arg("cross_device_semaphore"),
            py::arg("subdevice_id"),
            py::arg("cluster_axis"),
            py::arg("mesh_device"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = tt::tt_fabric::Topology::Linear,
            py::arg("queue_id") = DefaultQueueId,

        });
}

}  // namespace ttnn::operations::experimental::ccl
