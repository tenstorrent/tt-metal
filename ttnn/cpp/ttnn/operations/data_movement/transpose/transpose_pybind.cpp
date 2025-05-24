// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "transpose_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"

#include "transpose.hpp"

namespace ttnn::operations::data_movement::detail {
namespace py = pybind11;

void bind_transpose(py::module& module) {
    auto doc =
        R"doc(
            transpose(input_tensor: ttnn.Tensor, dim1: int, dim2: int, *, Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

            Returns a tensor that is transposed along dims dim1 and dim2

            Equivalent pytorch code:

            .. code-block:: python

                output_tensor = torch.transpose(input_tensor, 0, 1)

            Args:
                * :attr:`input_tensor`: Input Tensor.
                * :attr:`dim1`: First dim of transpose.
                * :attr:`dim2`: Second dim of transpose.
                * :attr:`pad_value` (Optional[float]): padding value for when tiles are broken in a transpose. Defaults to `0.0`. If set to None, it will be random garbage values.

            Keyword Args:
                * :attr:`memory_config`: Memory Config of the output tensor
                * :attr:`queue_id` (Optional[uint8]): command queue id
        )doc";

    using OperationType = decltype(ttnn::transpose);
    ttnn::bind_registered_operation(
        module,
        ttnn::transpose,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int64_t& dim1,
               const int64_t& dim2,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               QueueId queue_id,
               const std::optional<float>& pad_value) {
                return self(queue_id, input_tensor, dim1, dim2, memory_config, pad_value);
            },
            py::arg("input_tensor"),
            py::arg("dim1"),
            py::arg("dim2"),
            py::kw_only(),
            py::arg("memory_config") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId,
            py::arg("pad_value") = 0.0f,
        });
}
}  // namespace ttnn::operations::data_movement::detail
