// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "expand.hpp"
#include "expand_pybind.hpp"

namespace ttnn::operations::data_movement {
namespace py = pybind11;

namespace detail {
template <typename data_movement_operation_t>
void py_bind_expand(py::module& module, const data_movement_operation_t& operation, const char* doc) {
    ttnn::bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<int32_t> output_shape,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const QueueId queue_id) { return self(input_tensor, output_shape, memory_config, queue_id); },
            py::arg("input_tensor"),
            py::arg("output_shape"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace detail

void py_bind_expand(py::module& module) {
    auto doc =
        R"doc(expand(input: ttnn.Tensor, output_shape: List[int], memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor
        Returns a new tensor where singleton dimensions are expanded to a larger side.
        Unlike :func:`torch.expand`, this function is not zero-cost and perform a memory copy to create the expanded tensor. This is due to `ttnn.Tensor`'s lack of strided tensor support.

        Args:
            * :attr:`input`: The tensor to be expanded.
            * :attr:`output_shape`: The desired output shape.
            * :attr:`memory_config`: The memory configuration for the expanded tensor.

        Requirements:
            like torch.expand:
                only size 1 dimensions can be expanded in the output shape
                -1 or the original shape size can be used to indicate that dimension should not have an expansion
                The output shape must have the same or higher dimensions than the input shape

        )doc";

    detail::py_bind_expand(module, ttnn::expand, doc);
}

}  // namespace ttnn::operations::data_movement
