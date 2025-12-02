// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_reduce_pybind.hpp"

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>

#include "ttnn-pybind/decorators.hpp"
#include "all_reduce.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::ccl {

void py_bind_all_reduce(py::module& module) {
    const auto* doc =
        R"doc(
        All-reduce operation across devices with Sum reduction. If cluster axis is specified, the all-reduce is performed on tensor shards along the cluster axis, resulting in identical tensor shards across all devices along the cluster axis. If it is not specified, then we reduce across all devices in the mesh. All-reduce is a collective operation that reduces data from all devices using the Sum operation and returns the result to all devices.

        Args:
            input_tensor (ttnn.Tensor): Input tensor to be reduced.

        Keyword Args:
            cluster_axis (int, optional): The axis on the mesh device to reduce across. Defaults to `None`.
            subdevice_id (ttnn.SubDeviceId, optional): Subdevice id for worker cores.
            memory_config (ttnn.MemoryConfig, optional): Output memory configuration.
            num_links (int, optional): Number of links to use for the all_reduce operation. Defaults to `None`.
            topology (ttnn.Topology, optional): Fabric topology. Defaults to `None`.

        Returns:
            ttnn.Tensor: The reduced tensor with the same shape as the input tensor. The output tensor is identical across all devices along the cluster axis if specified, otherwise it is identical across all devices in the mesh.

        Example:
            >>> full_tensor = torch.randn([1, 1, 32, 256], dtype=torch.bfloat16)
            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
            >>> ttnn_tensor = ttnn.from_torch(
                            full_tensor,
                            dtype=input_dtype,
                            device=mesh_device,
                            layout=layout,
                            memory_config=mem_config,
                            mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(1, 8), dims=(-1, -2)))
            >>> output = ttnn.all_reduce(ttnn_tensor)
            >>> print(output.shape)
            [1, 1, 32, 256]
        )doc";

    using OperationType = decltype(ttnn::all_reduce);
    ttnn::bind_registered_operation(
        module,
        ttnn::all_reduce,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<uint32_t> cluster_axis,
               const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<uint32_t> num_links,
               const std::optional<tt::tt_fabric::Topology> topology) {
                return self(input_tensor, cluster_axis, subdevice_id, memory_config, num_links, topology);
            },
            py::arg("input_tensor").noconvert(),
            py::kw_only(),
            py::arg("cluster_axis") = std::nullopt,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("num_links") = std::nullopt,
            py::arg("topology") = std::nullopt,
        });
}

}  // namespace ttnn::operations::ccl
