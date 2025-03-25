// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>

#include "cpp/pybind11/decorators.hpp"

#include "ttnn/operations/experimental/reduction/sort/sort.hpp"
#include "ttnn/operations/experimental/reduction/sort/device/sort_device_operation.hpp"

namespace ttnn::operations::experimental::reduction::detail {
namespace py = pybind11;

void bind_reduction_sort_operation(py::module& module) {
    auto doc =
        R"doc(

            Sorts the elements of the ``input_tensor`` along a given dimension in ascending order by value.
            If no ``dim`` is provided, last dimension of the ``input`` is chosen.

            Equivalent pytorch code:

            .. code-block:: python

                return torch.sort(input_tensor, dim=-1)

            Args:
                * :attr:`input_tensor`: Input Tensor for sort.

            Keyword Args:
                * :attr:`dim`: the dimension to sort along (default to -1),
                * :attr:`descending`: sorting order - ascending or descending (default to False),
                * :attr:`stable`: whether to keep the original order of elements with equal values (default to False),
                * :attr:`memory_config`: Memory Config of the output tensor (default to False),
                * :attr:`output_tensor`: Preallocated output tensors (default to None).
        )doc";

    using OperationType = decltype(ttnn::experimental::sort);
    bind_registered_operation(
        module,
        ttnn::experimental::sort,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int8_t dim,
               const bool descending,
               const bool stable,
               std::optional<std::tuple<ttnn::Tensor, ttnn::Tensor>> optional_output_tensors,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) {
                return self(queue_id, input_tensor, dim, descending, stable, memory_config, optional_output_tensors);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("dim") = DIM_DEFAULT_VALUE,
            py::arg("descending") = DESCENDING_DEFAULT_VALUE,
            py::arg("stable") = STABLE_DEFAULT_VALUE,
            py::kw_only(),
            py::arg("out") = std::nullopt,
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimental::reduction::detail
