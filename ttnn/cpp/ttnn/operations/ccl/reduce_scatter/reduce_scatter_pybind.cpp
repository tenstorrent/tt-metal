// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
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
               const uint32_t scatter_dim,
               ttnn::operations::reduction::ReduceType math_op,
               const uint32_t num_links,
               const ttnn::MemoryConfig& memory_config,
               const std::optional<size_t> num_workers,
               const std::optional<size_t> num_buffers_per_channel) -> ttnn::Tensor {
                return self(input_tensor, scatter_dim, math_op, num_links, memory_config, num_workers, num_buffers_per_channel);
            },
            py::arg("input_tensor"),
            py::arg("scatter_dim"),
            py::arg("math_op"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("num_workers") = std::nullopt,
            py::arg("num_buffers_per_channel") = std::nullopt});
}

}  // namespace detail


void py_bind_reduce_scatter(pybind11::module& module) {

    detail::bind_reduce_scatter(
        module,
        ttnn::reduce_scatter,
        R"doc(reduce_scatter(input_tensor: std::vector<ttnn.Tensor>, scatter_dim: int, math_op: ReduceType, *, num_links: int = 1, memory_config: Optional[ttnn.MemoryConfig] = None, num_workers: int = None, num_buffers_per_channel: int = None) -> std::vector<ttnn.Tensor>

        Performs an reduce_scatter operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`dim` (int)

        Keyword Args:
            * :attr:`num_links` (int): Number of links to use for the all-gather operation.
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
            * :attr:`num_workers` (int): Number of workers to use for the operation.
            * :attr:`num_buffers_per_channel` (int): Number of buffers per channel to use for the operation.

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.reduce_scatter(tensor, dim=0)

        )doc");
}

}  // namespace ttnn::operations::ccl
