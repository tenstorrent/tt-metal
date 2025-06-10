// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "repeat.hpp"

namespace ttnn::operations::data_movement {
namespace py = pybind11;

namespace detail {
template <typename data_movement_operation_t>
void bind_repeat(py::module& module, const data_movement_operation_t& operation, const char* doc) {
    ttnn::bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<uint32_t>& repetition_vector,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) { return self(input_tensor, repetition_vector, memory_config, queue_id); },
            py::arg("input_tensor"),
            py::arg("repeat_dims"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}

}  // namespace detail

void py_bind_repeat(py::module& module) {
    auto doc = R"doc(

    Returns a new tensor filled with repetition of input :attr:`input_tensor` according to number of times specified in :attr:`shape`.

    Args:
        input_tensor (ttnn.Tensor): the input tensor.
        repetition_vector (SmallVector): The number of repetitions for each dimension.

    Keyword Args:
        memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

    Returns:
        ttnn.Tensor: the output tensor.

    Example:

        >>> tensor = ttnn.repeat(ttnn.from_torch(torch.tensor([[1, 2], [3, 4]]), [1,2],)), device)
        >>> print(tensor)
        tensor([[1, 2],
        [1, 2],
        [3, 4],
        [3, 4]])
            )doc";

    detail::bind_repeat(module, ttnn::repeat, doc);
}

}  // namespace ttnn::operations::data_movement
