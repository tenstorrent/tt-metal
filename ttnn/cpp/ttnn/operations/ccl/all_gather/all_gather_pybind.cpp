// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::ccl {

namespace detail {

template <typename ccl_operation_t>
void bind_all_gather(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    py::enum_<ttnn::ccl::Topology>(module, "Topology")
        .value("Ring", ttnn::ccl::Topology::Ring)
        .value("Linear", ttnn::ccl::Topology::Linear);

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const uint32_t dim,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<size_t> num_workers,
               const std::optional<size_t> num_buffers_per_channel,
               const ttnn::ccl::Topology topology) -> ttnn::Tensor {
                return self(
                    input_tensor, dim, num_links, memory_config, num_workers, num_buffers_per_channel, topology);
            },
            py::arg("input_tensor"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("num_workers") = std::nullopt,
            py::arg("num_buffers_per_channel") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring},

        ttnn::pybind_overload_t{[](const ccl_operation_t& self,
                                   const ttnn::Tensor& input_tensor,
                                   const uint32_t dim,
                                   const uint32_t cluster_axis,
                                   const MeshDevice& mesh_device,
                                   const uint32_t num_links,
                                   const std::optional<ttnn::MemoryConfig>& memory_config,
                                   const std::optional<size_t> num_workers,
                                   const std::optional<size_t> num_buffers_per_channel,
                                   const ttnn::ccl::Topology topology) -> ttnn::Tensor {
                                    return self(input_tensor,
                                                dim,
                                                cluster_axis,
                                                mesh_device,
                                                num_links,
                                                memory_config,
                                                num_workers,
                                                num_buffers_per_channel,
                                                topology);
                                },
                                py::arg("input_tensor"),
                                py::arg("dim"),
                                py::arg("cluster_axis"),
                                py::arg("mesh_device"),
                                py::kw_only(),
                                py::arg("num_links") = 1,
                                py::arg("memory_config") = std::nullopt,
                                py::arg("num_workers") = std::nullopt,
                                py::arg("num_buffers_per_channel") = std::nullopt,
                                py::arg("topology") = ttnn::ccl::Topology::Ring});
}

}  // namespace detail

void py_bind_all_gather(pybind11::module& module) {
    detail::bind_all_gather(module,
                            ttnn::all_gather,
                            R"doc(

        Performs an all-gather operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor.
            dim (int): Dimension to perform operation.
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the line-all-gather operation on.
            mesh_device (MeshDevice): Device mesh to perform the line-all-gather operation on.
        * cluster_axis and mesh_device parameters are applicable only for Linear Topology.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming%20Mesh%20of%20Devices/Programming%20Mesh%20of%20Devices%20with%20TT-NN.md

        Keyword Args:
            num_links (int, optional): Number of links to use for the all-gather operation. Defaults to `1`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            num_workers (int, optional): Number of workers to use for the operation. Defaults to `None`.
            num_buffers_per_channel (int, optional): Number of buffers per channel to use for the operation. Defaults to `None`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Ring`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> full_tensor = torch.randn([1, 1, 32, 256], dtype=torch.bfloat16)
            >>> physical_device_ids = ttnn.get_t3k_physical_device_ids_ring()
            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), physical_device_ids=physical_device_ids[:8])
            >>> ttnn_tensor = ttnn.from_torch(
                            full_tensor,
                            dtype=input_dtype,
                            device=mesh_device,
                            layout=layout,
                            memory_config=mem_config,
                            mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(1, 8), dims=(-1, -2)))
            >>> ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
            >>> output = ttnn.all_gather(ttnn_tensor, dim=0, topology=ttnn.Topology.Ring)

        )doc");
}

}  // namespace ttnn::operations::ccl
