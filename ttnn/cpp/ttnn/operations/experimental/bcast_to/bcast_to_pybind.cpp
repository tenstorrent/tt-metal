// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cpp/pybind11/decorators.hpp"

#include "bcast_to_pybind.hpp"
#include "ttnn/operations/experimental/bcast_to/bcast_to.hpp"

namespace ttnn::operations::experimental::broadcast_to::detail {
namespace py = pybind11;

void py_bind_broadcast_to(py::module& module) {
    const auto* doc =
        R"doc(broadcast_to(input: ttnn.Tensor, sizes: List[int], output: Optional[ttnn.Tensor] = None, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor
        Returns a new tensor where singleton dimensions are expanded to given sizes.
        Unlike :func:`torch.broadcast_to`, this function is not zero-cost and perform a memory copy to create the expanded tensor. This is due to `ttnn.Tensor`'s lack of strided tensor support.

        Args:
            * :attr:`input`: The tensor to be broadcasted.
            * :attr:`sizes`: The desired broadcasted size.
            * :attr:`output`: An optional tensor to store the broadcasted result.
            * :attr:`memory_config`: The memory configuration for the broadcasted tensor.
        )doc";
    using operationType = decltype(ttnn::experimental::broadcast_to);
    bind_registered_operation(
        module,
        ttnn::experimental::broadcast_to,
        doc,
        ttnn::pybind_overload_t{
            [](const operationType& self,
               const ttnn::Tensor& input,
               const std::vector<int32_t>& sizes,
               const std::optional<ttnn::Tensor>& output_tensor,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) -> ttnn::Tensor { return self(queue_id, input, sizes, output_tensor, memory_config); },
            py::arg("input"),
            py::arg("sizes"),
            py::kw_only(),
            py::arg("output") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}
}  // namespace ttnn::operations::experimental::broadcast_to::detail
