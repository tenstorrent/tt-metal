// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/ccl/line_all_gather/line_all_gather.hpp"
#include "ttnn/types.hpp"


namespace ttnn::operations::ccl {

namespace detail {

template <typename ccl_operation_t>
void bind_line_all_gather(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
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
               const std::optional<size_t> num_buffers_per_channel) -> ttnn::Tensor {
                return self(input_tensor, dim, num_links, memory_config, num_workers, num_buffers_per_channel);
            },
            py::arg("input_tensor"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("num_workers") = std::nullopt,
            py::arg("num_buffers_per_channel") = 2},
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const uint32_t dim,
               const uint32_t cluster_axis,
               const MeshDevice& mesh_device,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<size_t> num_workers,
               const std::optional<size_t> num_buffers_per_channel) -> ttnn::Tensor {
                return self(input_tensor, dim, cluster_axis, mesh_device, num_links, memory_config, num_workers, num_buffers_per_channel);
            },
            py::arg("input_tensor"),
            py::arg("dim"),
            py::arg("cluster_axis"),
            py::arg("mesh_device"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("num_workers") = std::nullopt,
            py::arg("num_buffers_per_channel") = 2});
}

}  // namespace detail


void py_bind_line_all_gather(pybind11::module& module) {

    detail::bind_line_all_gather(
        module,
        ttnn::line_all_gather,
        R"doc(line_all_gather(input_tensor: ttnn.Tensor, dim: int, *, num_links: int = 1, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

        Performs an all-gather operation on multi-device :attr:`input_tensor` across all devices.
        Args:
            * :attr:`input_tensor` (ttnn.Tensor):
                multi-device tensor
            * :attr:`dim` (int):
                Dimension to perform the all-gather operation on.
                After the all-gather operation, the size of the :attr:`dim`
                dimension will larger by number of devices in the line.
            * :attr:`cluster_axis` (int):
                Provided a MeshTensor, the axis corresponding to MeshDevice
                to perform the line-all-gather operation on.
            * :attr:`mesh_device` (MeshDevice):
                Device mesh to perform the line-all-gather operation on.

        Keyword Args:
            * :attr:`num_links` (int): Number of links to use for the all-gather operation.
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.line_all_gather(tensor, dim=0)

        )doc");
}

}  // namespace ttnn::operations::ccl
