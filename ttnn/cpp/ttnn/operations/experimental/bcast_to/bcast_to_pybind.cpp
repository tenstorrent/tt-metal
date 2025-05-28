// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_to_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ttnn-pybind/decorators.hpp"

#include "ttnn/operations/experimental/bcast_to/bcast_to.hpp"

namespace ttnn::operations::experimental::broadcast_to::detail {
namespace py = pybind11;

void py_bind_broadcast_to(py::module& module) {
    const auto* doc =
        R"doc(broadcast_to(input: ttnn.Tensor, ttnn::Shape, memory_config: Optional[ttnn.MemoryConfig] = None, output: Optional[ttnn.Tensor] = None) -> ttnn.Tensor
        Returns a new tensor where singleton dimensions are broadcasted to the given shape.

        Args:
            * :attr:`input`: The tensor to be broadcasted.
            * :attr:`output_shape`: The desired broadcasted shape.
            * :attr:`memory_config`: The memory configuration for the broadcasted tensor.
            * :attr:`output`: An optional tensor to store the broadcasted result.

        Notes:
            Currently Supports:
            Data Type
	            bfloat16
	            float32

            Tensor Shape
                up to 4D

            Memory Layout
                Tile

            Memory Config
                Interleaved (DRAM / L1)
        )doc";
    using operationType = decltype(ttnn::experimental::broadcast_to);
    bind_registered_operation(
        module,
        ttnn::experimental::broadcast_to,
        doc,
        ttnn::pybind_overload_t{
            [](const operationType& self,
               const ttnn::Tensor& input,
               const ttnn::Shape& output_shape,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::optional<ttnn::Tensor>& output_tensor,
               QueueId queue_id) -> ttnn::Tensor {
                return self(queue_id, input, output_shape, memory_config, output_tensor);
            },
            py::arg("input"),
            py::arg("output_shape"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("output") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}
}  // namespace ttnn::operations::experimental::broadcast_to::detail
