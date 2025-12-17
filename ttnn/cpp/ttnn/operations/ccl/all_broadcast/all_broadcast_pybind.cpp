// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_broadcast_pybind.hpp"

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>

#include "ttnn-pybind/decorators.hpp"
#include "all_broadcast.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn::operations::ccl {

void py_bind_all_broadcast(py::module& module) {
    const auto* doc =
        R"doc(all_broadcast(input_tensor: ttnn.Tensor, *, cluster_axis: Optional[int] = None, subdevice_id: Optional[ttnn.SubDeviceId] = None, memory_config: Optional[ttnn.MemoryConfig] = None, num_links: Optional[int] = 1, topology: Optional[ttnn.Topology] = ttnn.Topology.Linear) -> List[ttnn.Tensor]

            All-broadcast operation across devices. This operation broadcasts data from all devices to all other devices in the mesh, returning a vector of tensors where each tensor contains the data from a corresponding device in the mesh. The output tensors are identical across all devices along the cluster axis if specified, otherwise they are identical across all devices in the mesh.

            Args:
                input_tensor (ttnn.Tensor): Input tensor to be broadcast.

            Keyword Args:
                cluster_axis (int, optional): The axis on the mesh device to broadcast across. Defaults to `None`.
                subdevice_id (ttnn.SubDeviceId, optional): Subdevice id for worker cores.
                memory_config (ttnn.MemoryConfig, optional): Output memory configuration. Defaults to input tensor memory config.
                num_links (int, optional): The number of links to use for the all-broadcast operation. Defaults to `1`.
                topology (ttnn.Topology, optional): Fabric topology. Defaults to `ttnn.Topology.Linear`.

           Returns:
               List[ttnn.Tensor]: A list of tensors, one from each device, where each tensor has the same shape as the input.

            Example:
                >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
                >>> input_tensor = ttnn.from_torch(
                                torch.rand([1, 1, 32, 256]),
                                dtype=ttnn.bfloat16,
                                device=mesh_device,
                                layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
                >>> output = ttnn.all_broadcast(input_tensor)
                >>> # output is a list of 8 tensors, each with shape [1, 1, 32, 256]
                )doc";

    using OperationType = decltype(ttnn::all_broadcast);
    ttnn::bind_registered_operation(
        module,
        ttnn::all_broadcast,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<uint32_t> cluster_axis,
               const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<uint32_t> num_links,
               const std::optional<ttnn::ccl::Topology> topology) {
                return self(input_tensor, cluster_axis, subdevice_id, memory_config, num_links, topology);
            },
            py::arg("input_tensor").noconvert(),
            py::kw_only(),
            py::arg("cluster_axis") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("num_links") = 1,
            py::arg("topology") = ttnn::ccl::Topology::Linear,
        });
}

}  // namespace ttnn::operations::ccl
