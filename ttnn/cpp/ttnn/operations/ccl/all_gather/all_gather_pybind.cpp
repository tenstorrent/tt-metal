// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/types.hpp"

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
                return self(input_tensor, dim, num_links, memory_config, num_workers, num_buffers_per_channel, topology);
            },
            py::arg("input_tensor"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("num_workers") = std::nullopt,
            py::arg("num_buffers_per_channel") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring},

        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const uint32_t dim,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<size_t> num_workers,
               const std::optional<size_t> num_buffers_per_channel,
               const ttnn::ccl::Topology topology) -> ttnn::Tensor {
                return self(input_tensor, dim, cluster_axis, mesh_device, num_links, memory_config, num_workers, num_buffers_per_channel, topology);
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
    detail::bind_all_gather(
        module,
        ttnn::all_gather,
        R"doc(all_gather(input_tensor: ttnn.Tensor, dim: int, *, num_links: int = 1, memory_config: Optional[ttnn.MemoryConfig] = None, num_workers: int = None, num_buffers_per_channel: int = None, Topology: ttnn.Topology = ttnn.Topology.Ring) -> ttnn.Tensor

        Performs an all-gather operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`dim` (int)
            * Following are applicable only for Linear Topology
            * :attr:`cluster_axis` (int):
                Provided a MeshTensor, the axis corresponding to MeshDevice
                to perform the line-all-gather operation on.
            * :attr:`mesh_device` (MeshDevice):
                Device mesh to perform the line-all-gather operation on.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming%20Mesh%20of%20Devices/Programming%20Mesh%20of%20Devices%20with%20TT-NN.md

        Keyword Args:
            * :attr:`num_links` (int): Number of links to use for the all-gather operation.
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
            * :attr:`num_workers` (int): Number of workers to use for the operation.
            * :attr:`num_buffers_per_channel` (int): Number of buffers per channel to use for the operation.
            * :attr:`topology`: Topology to be used for the operation. Allowable options are Linear and Ring

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.all_gather(tensor, dim=0)

        )doc");
}

}  // namespace ttnn::operations::ccl
