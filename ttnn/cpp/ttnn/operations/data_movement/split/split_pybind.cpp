// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "split_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "split.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_split(py::module& module) {
    auto doc =
        R"doc(
            split(input_tensor: ttnn.Tensor, num_splits: int, dim: int, *, Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Returns a tensor that is in num_splits ways on dim.

            Equivalent pytorch code:

            .. code-block:: python

                output_tensor = torch.split(input_tensor, 2, 1)

            Args:
                * :attr:`input_tensor`: Input Tensor.
                * :attr:`split_size` (Union[int, list[int]]): Single chunk size or list of chunk sizes. Output may be smaller if dim not evenly divisible.
                * :attr:`dim2`: Dim to split. Defaults to 0.

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id
        )doc";

    using OperationType = decltype(ttnn::split);
    ttnn::bind_registered_operation(
        module,
        ttnn::split,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int64_t split_size,
               const int64_t dim,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id) { return self(queue_id, input_tensor, split_size, dim, memory_config); },
            py::arg("input_tensor"),
            py::arg("split_size"),
            py::arg("dim") = 0,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = 0,
        },
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const ttnn::SmallVector<int64_t>& split_sizes,
               const int64_t dim,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const QueueId queue_id) { return self(queue_id, input_tensor, split_sizes, dim, memory_config); },
            py::arg("input_tensor"),
            py::arg("split_size"),
            py::arg("dim") = 0,
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
        });
}
}  // namespace ttnn::operations::data_movement::detail
