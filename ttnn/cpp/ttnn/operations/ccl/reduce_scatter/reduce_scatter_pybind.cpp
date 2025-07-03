// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn::operations::ccl {

namespace detail {

template <typename ccl_operation_t>
void bind_reduce_scatter(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const int32_t dim,
               ttnn::operations::reduction::ReduceType math_op,
               const uint32_t num_links,
               const ttnn::MemoryConfig& memory_config,
               ttnn::ccl::Topology topology,
               const std::optional<size_t> num_workers,
               const std::optional<size_t> num_buffers_per_channel) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    dim,
                    math_op,
                    num_links,
                    memory_config,
                    topology,
                    num_workers,
                    num_buffers_per_channel);
            },
            py::arg("input_tensor"),
            py::arg("dim"),
            py::arg("math_op"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring,
            py::arg("num_workers") = std::nullopt,
            py::arg("num_buffers_per_channel") = std::nullopt},

        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const int32_t dim,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               ttnn::operations::reduction::ReduceType math_op,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& output_mem_config,
               const std::optional<size_t> num_workers,
               const std::optional<size_t> num_buffers_per_channel,
               const ttnn::ccl::Topology topology) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    dim,
                    cluster_axis,
                    mesh_device,
                    math_op,
                    num_links,
                    output_mem_config,
                    topology,
                    num_workers,
                    num_buffers_per_channel);
            },
            py::arg("input_tensor"),
            py::arg("dim"),
            py::arg("cluster_axis"),
            py::arg("mesh_device"),
            py::arg("math_op"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("num_workers") = std::nullopt,
            py::arg("num_buffers_per_channel") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring});
}

}  // namespace detail

void py_bind_reduce_scatter(pybind11::module& module) {
    detail::bind_reduce_scatter(
        module,
        ttnn::reduce_scatter,
        R"doc(

        Performs an reduce_scatter operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor
            dim (int): Dimension to perform operation
            cluster_axis (int): Provided a MeshTensor, the axis corresponding to MeshDevice to perform the line-reduce-scatter operation on.
            mesh_device (MeshDevice): Device mesh to perform the line-reduce-scatter operation on.
        * cluster_axis and mesh_device parameters are applicable only for Linear Topology.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming%20Mesh%20of%20Devices/Programming%20Mesh%20of%20Devices%20with%20TT-NN.md

        Keyword Args:
            num_links (int, optional): Number of links to use for the reduce0scatter operation. Defaults to `1`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            num_workers (int, optional): Number of workers to use for the operation. Defaults to `None`.
            num_buffers_per_channel (int, optional): Number of buffers per channel to use for the operation. Defaults to `None`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Ring`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:

            >>> full_tensor = torch.randn([1, 1, 256, 256], dtype=torch.bfloat16)
            >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
            >>> input_tensor = ttnn.from_torch(
                    full_tensor,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
                )
            >>> output = ttnn.reduce_scatter(input_tensor, dim=0, topology=ttnn.Topology.Linear)

        )doc");
}

}  // namespace ttnn::operations::ccl
