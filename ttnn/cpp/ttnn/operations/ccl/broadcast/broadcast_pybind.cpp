// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "broadcast_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/ccl/broadcast/broadcast.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::ccl {

namespace {

template <typename ccl_operation_t>
void bind_broadcast(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const MeshCoordinate& sender_coord,
               std::optional<uint32_t> cluster_axis,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const uint32_t num_links,
               const ttnn::ccl::Topology topology) -> ttnn::Tensor {
                return self(input_tensor, sender_coord, num_links, memory_config, topology, cluster_axis, subdevice_id);
            },
            py::arg("input_tensor"),
            py::arg("sender_coord"),
            py::kw_only(),
            py::arg("cluster_axis") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("num_links") = 1,
            py::arg("topology") = ttnn::ccl::Topology::Linear});
}

}  // namespace

void py_bind_broadcast(pybind11::module& module) {
    bind_broadcast(
        module,
        ttnn::broadcast,
        R"doc(
        Performs a broadcast operation from a sender device to all other mesh devices across a cluster axis.

        Args:
            input_tensor (ttnn.Tensor)
            sender_coord (MeshCoordinate): Coordinate of the sender device in the mesh.
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the operation on.
            mesh_device (MeshDevice): Device mesh to perform the operation on.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Keyword Args:
            num_links (int, optional): Number of links to use for the all-broadcast operation. Defaults to `1`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Ring`.

        Returns:
            ttnn.Tensor of the output on the mesh device.

        Example:
            >>> sender_tensor = torch.randn([1, 1, 32, 256], dtype=torch.bfloat16)
            >>> num_devices = 4
            >>> device_tensors = []
            >>> for device_idx in range(num_devices):
                    if device_idx == sender_coord_tuple[cluster_axis]:
                        device_tensors.append(sender_tensor)
                    else:
                        device_tensors.append(torch.zeros_like(sender_tensor))
            >>> mesh_tensor_torch = torch.cat(device_tensors, dim=-1)
            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
            >>> sender_coord = MeshCoordinate((0, 0))
            >>> mesh_mapper_config = ttnn.MeshMapperConfig(
                    [ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)], ttnn.MeshShape(1, num_devices)
                )
            >>> ttnn_tensor = ttnn.from_torch(
                            mesh_tensor_torch,
                            dtype=input_dtype,
                            device=mesh_device,
                            layout=layout,
                            memory_config=mem_config,
                            mesh_mapper=ttnn.create_mesh_mapper(mesh_device,mesh_mapper_config))
            >>> output = ttnn.broadcast(ttnn_tensor, sender_coord)

        )doc");
}

}  // namespace ttnn::operations::ccl
